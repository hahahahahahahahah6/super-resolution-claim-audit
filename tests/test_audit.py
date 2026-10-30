import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from super_resolution_claim_audit.audit import audit, claims, evidence, main, matches
from super_resolution_claim_audit.phrases import SUPERRESOLUTION, NOSUPERRESOLUTION

ROOT = Path(__file__).resolve().parents[1]

class Parsing(unittest.TestCase):
    def test_fences(self):
        for fence in ('```', '~~~', '   ````'):
            self.assertEqual(claims(fence+'\nsuper resolution\n'+fence), (False, False))
    def test_inline(self):
        self.assertEqual(claims('`super resolution`'), (True, False))
    def test_negative(self):
        self.assertEqual(claims('does not support image super resolution'), (False, True))
    def test_hyphens(self):
        self.assertEqual(claims('super-resolution'), (True, False))
    def test_contraction(self):
        self.assertEqual(claims("doesn't support image super resolution"), (False, True))
    def test_join(self):
        self.assertEqual(claims('super\nresolution'), (True, False))
        self.assertEqual(claims('super\n\nresolution'), (False, False))
    def test_longest(self):
        self.assertEqual(matches('image super resolution', SUPERRESOLUTION), [(0,22)])
    def test_boundaries(self):
        self.assertEqual(claims('supersuper resolutions'), (False, False))
    def test_tuples(self):
        for phrases in (SUPERRESOLUTION, NOSUPERRESOLUTION):
            self.assertIsInstance(phrases, tuple)
            self.assertEqual(list(phrases), sorted(phrases,key=len,reverse=True))
            self.assertEqual(len(phrases),len(set(phrases)))
    def test_all_phrases(self):
        for phrase in SUPERRESOLUTION:
            self.assertEqual(claims(phrase),(True,False),phrase)
        for phrase in NOSUPERRESOLUTION:
            self.assertEqual(claims(phrase),(False,True),phrase)

class Evidence(unittest.TestCase):
    pass

positive = {
 'opencv_direct':'import cv2\ncv2.dnn_superres.DnnSuperResImpl_create()',
 'opencv_alias':'import cv2 as c\nc.dnn_superres.DnnSuperResImpl_create()',
 'opencv_from':'from cv2.dnn_superres import DnnSuperResImpl_create as make\nmake()',
 'opencv_namespace':'from cv2 import dnn_superres as sr\nsr.DnnSuperResImpl_create()',
 'real_direct':'import realesrgan\nrealesrgan.RealESRGANer()',
 'real_from':'from realesrgan import RealESRGANer\nRealESRGANer()',
 'rrdb':'from basicsr.archs.rrdbnet_arch import RRDBNet\nRRDBNet()',
 'swinir':'from models.network_swinir import SwinIR\nSwinIR()',
 'pipeline_pos':'from transformers import pipeline as p\np("image-super-resolution")',
 'pipeline_kw':'import transformers as t\nt.pipeline(task="SUPER-RESOLUTION")',
}
for model in ('EDSR','RCAN','SRResNet','LapSRN','FSRCNN'):
 positive[model]='import models\nmodels.'+model+'()'
negative = {
 'bare':'from realesrgan import RealESRGANer',
 'pil':'from PIL import Image\nImage.resize((2,2))',
 'resize':'import cv2\ncv2.resize(img,(2,2))',
 'interpolate':'import torch\ntorch.nn.functional.interpolate(x)',
 'video_resize':'import ffmpeg\nffmpeg.filter(x,"scale",20,20)',
 'crop':'image.crop((0,0,10,10))',
 'sharpen':'from PIL import ImageFilter\nimage.filter(ImageFilter.SHARPEN)',
 'denoise':'import cv2\ncv2.fastNlMeansDenoising(x)',
 'convert':'image.convert("RGB")',
 'detection':'from ultralytics import YOLO\nYOLO("model").predict(img)',
 'segmentation':'from mediapipe import SelfieSegmentation\nSelfieSegmentation()',
 'classification':'from transformers import pipeline\npipeline("image-classification")',
 'face':'import cv2\ncv2.FaceDetectorYN_create()',
 'hand':'from mediapipe import Hands\nHands()',
 'pose':'from mediapipe import Pose\nPose()',
 'tracking':'import cv2\ncv2.TrackerCSRT_create()',
 'depth':'from transformers import pipeline\npipeline("depth-estimation")',
 'flow':'import cv2\ncv2.calcOpticalFlowFarneback(a,b)',
 'ocr':'import easyocr\neasyocr.Reader(["en"])',
 'barcode':'from pyzbar.pyzbar import decode\ndecode(img)',
 'difference':'import cv2\ncv2.absdiff(a,b)',
 'writer':'import cv2\ncv2.VideoWriter()',
 'kalman':'import cv2\ncv2.KalmanFilter()',
 'shadow':'from realesrgan import RealESRGANer\ndef RealESRGANer(): pass\nRealESRGANer()',
 'class_shadow':'from models import EDSR\nclass EDSR: pass\nEDSR()',
 'assignment_shadow':'import cv2\ncv2 = fake\ncv2.dnn_superres.DnnSuperResImpl_create()',
 'string':'x="super resolution"',
 'subprocess':'import subprocess\nsubprocess.run(["RealESRGANer"])',
 'invalid':'from realesrgan import RealESRGANer\nRealESRGANer((',
 'unimported':'RealESRGANer()',
 'dynamic_task':'from transformers import pipeline\npipeline(task=task)',
 'other_pipeline':'from mylib import pipeline\npipeline("super-resolution")',
}
def make_evidence(source, expected):
    def test(self): self.assertEqual(evidence(source),expected)
    return test
for name,source in positive.items(): setattr(Evidence,'test_positive_'+name,make_evidence(source,True))
for name,source in negative.items(): setattr(Evidence,'test_negative_'+name,make_evidence(source,False))

class Integration(unittest.TestCase):
    def cli(self,*args):
        output=io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try: code=main(list(args))
            except SystemExit as e: code=e.code
        return code,output.getvalue()
    def test_good(self):
        self.assertEqual(audit(ROOT/'fixtures/good')['findings'],[])
    def test_bad(self):
        code,text=self.cli(str(ROOT/'fixtures/bad'),'--json')
        self.assertEqual(code,2)
        result=json.loads(text)
        self.assertEqual(set(result),{'verdict','findings','files_scanned'})
        self.assertEqual([(f['severity'],f['rule']) for f in result['findings']], [('HIGH','documented-super-resolution-claim-false')])
    def test_testonly(self):
        for threshold,expected in [('HIGH',0),('MEDIUM',2)]:
            code,text=self.cli(str(ROOT/'fixtures/testonly'),'--json','--fail-on',threshold)
            self.assertEqual(code,expected)
            self.assertEqual([f['severity'] for f in json.loads(text)['findings']],['MEDIUM'])
    def test_no(self):
        code,text=self.cli(str(ROOT/'fixtures/no-super-resolution'),'--json')
        self.assertEqual(code,2)
        self.assertEqual([(f['severity'],f['rule']) for f in json.loads(text)['findings']], [('HIGH','documented-no-super-resolution-claim-false')])
    def test_ignore(self):
        self.assertEqual(audit(ROOT/'fixtures/bad',['README.*'])['findings'],[])
    def test_verbose(self):
        code,text=self.cli(str(ROOT/'fixtures/good'),'--verbose')
        self.assertEqual(code,0)
        self.assertIn('Files scanned: 2',text)
    def test_version(self): self.assertEqual(self.cli('--version'),(0,'0.1.0\n'))
    def test_invalid(self): self.assertEqual(self.cli('/missing/root')[0],1)
    def test_usage(self):
        self.assertEqual(self.cli()[0],1)
        self.assertEqual(self.cli('.','--fail-on','LOW')[0],1)
    def test_self(self): self.assertEqual(audit(ROOT)['findings'],[])
    def test_no_claim(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'main.py').write_text(positive['real_from'])
            self.assertEqual(audit(d)['findings'],[])
    def test_scopes(self):
        for target,expected in [('fixtures/m.py','HIGH'),('docs/m.py','HIGH'),('examples/m.py','HIGH'),('tests/m.py','MEDIUM'),('test_m.py','MEDIUM'),('m_test.py','MEDIUM'),('main.py',None)]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                root=Path(d); (root/'README.md').write_text('super resolution')
                file=root/target; file.parent.mkdir(parents=True,exist_ok=True); file.write_text(positive['real_from'])
                findings=audit(root)['findings']
                self.assertEqual([f['severity'] for f in findings],[] if expected is None else [expected])
    def test_docs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'nested/docs/deep').mkdir(parents=True)
            (root/'nested/docs/deep/a.md').write_text('super resolution')
            self.assertEqual(len(audit(root)['findings']),1)
    def test_symlinks(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as external:
            root=Path(d); (root/'README.md').write_text('super resolution')
            real=Path(external,'model.py'); real.write_text(positive['real_from'])
            (root/'linked.py').symlink_to(real)
            (root/'linked_dir').symlink_to(external,target_is_directory=True)
            self.assertEqual(audit(root)['findings'][0]['severity'],'HIGH')
