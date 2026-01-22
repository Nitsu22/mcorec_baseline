import importlib.util
from pathlib import Path

_impl_path = Path(__file__).with_name("tf-locoformer.py")
_spec = importlib.util.spec_from_file_location("tf_locoformer_impl", _impl_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

tf_locoformer_separator = _module.tf_locoformer_separator
TFLocoformerSeparator = _module.TFLocoformerSeparator

