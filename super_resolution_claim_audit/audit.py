"""Static documentation and source analysis."""
import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path
import re
from . import __version__
from .phrases import SUPERRESOLUTION, NOSUPERRESOLUTION


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def matches(text, phrases):
    spans = []
    for phrase in sorted(phrases, key=len, reverse=True):
        for match in re.finditer(r'\b' + re.escape(phrase) + r'\b', text):
            if not any(match.start() < b and match.end() > a for a, b in spans):
                spans.append(match.span())
    return spans


def claims(text):
    lines = []
    fence = None
    for line in text.splitlines():
        opening = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if re.match(r'^ {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}\s*$', line):
                fence = None
            continue
        if opening:
            fence = opening[1]
            lines.append('')
        else:
            lines.append(line)
    positive = negative = False
    for paragraph in re.split(r'\n\s*\n', '\n'.join(lines)):
        text = normalize(paragraph)
        spans = matches(text, NOSUPERRESOLUTION)
        negative |= bool(spans)
        for a, b in sorted(spans, reverse=True):
            text = text[:a] + ' ' * (b-a) + text[b:]
        positive |= bool(matches(text, SUPERRESOLUTION))
    return positive, negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    aliases = {}
    shadow = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = node.module + '.' + item.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadow.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            shadow.add(node.id)
        elif isinstance(node, ast.arg):
            shadow.add(node.arg)
    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id) if node.id not in shadow else None
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else None
        return None
    models = {'DnnSuperResImpl_create', 'RealESRGANer', 'RRDBNet', 'SwinIR', 'EDSR', 'RCAN', 'SRResNet', 'LapSRN', 'FSRCNN'}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = resolve(node.func)
        if not resolved:
            continue
        leaf = resolved.rsplit('.', 1)[-1]
        if leaf in models:
            return True
        if leaf == 'pipeline' and resolved.startswith('transformers.'):
            tasks = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == 'task']
            if any(isinstance(t, ast.Constant) and isinstance(t.value, str) and 'super-resolution' in t.value.lower() for t in tasks):
                return True
    return False


def audit(root, ignore=(), fail_on='HIGH'):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('invalid or unreadable root')
    positive = negative = production = testing = False
    count = 0
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not (Path(directory)/d).is_symlink())
        for filename in sorted(files):
            path = Path(directory)/filename
            rel = path.relative_to(root)
            if path.is_symlink() or any(fnmatch.fnmatchcase(rel.as_posix(), pattern) for pattern in ignore):
                continue
            parts = rel.parts
            doc = rel.as_posix() == 'README.md' or (path.suffix == '.md' and 'docs' in parts[:-1])
            source = path.suffix == '.py' and not set(parts[:-1]) & {'fixtures', 'docs', 'examples'}
            if not (doc or source):
                continue
            text = path.read_text(encoding='utf-8')
            count += 1
            if doc:
                pos, neg = claims(text)
                positive |= pos
                negative |= neg
            if source and evidence(text):
                test = 'tests' in parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
                testing |= test
                production |= not test
    findings = []
    if positive and not production:
        findings.append(dict(rule='documented-super-resolution-claim-false', severity='MEDIUM' if testing else 'HIGH', message='Documentation promises learned enlargement without production source evidence.'))
    if negative and production:
        findings.append(dict(rule='documented-no-super-resolution-claim-false', severity='HIGH', message='Documentation denies learned enlargement but production source evidence exists.'))
    failed = any(f['severity'] == 'HIGH' or fail_on == 'MEDIUM' for f in findings)
    return dict(verdict='FAIL' if failed else 'PASS', findings=findings, files_scanned=count)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage()
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser()
    parser.add_argument('root')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore', action='append', default=[])
    parser.add_argument('--fail-on', choices=['HIGH', 'MEDIUM'], default='HIGH')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        result = audit(args.root, args.ignore, args.fail_on)
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, str(error) + '\n')
    if args.json:
        print(json.dumps(result))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{severity} {rule}: {message}'.format(**finding))
        if args.verbose:
            print('Files scanned: ' + str(result['files_scanned']))
    return 2 if result['verdict'] == 'FAIL' else 0
