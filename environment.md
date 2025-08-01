# Environment setup

To get start:
```bash
cd ~
mkdir venv
```

## vllm-v0-jenga
```bash
python3 -m venv venv/vllm-v0-jenga
source venv/vllm-v0-jenga/bin/activate
git clone https://github.com/heheda12345/Jenga-SOSP25-AE --branch vllm-v0-jenga
cd vllm-v0-jenga
pip install -e .
pip uninstall transformers
pip install transformers==4.45.2
pip install sortedcontainers
deactivate
```

## vllm-v0-baseline
```bash
cd ~
python3 -m venv venv/vllm-v0-baseline
source venv/vllm-v0-baseline/bin/activate
git clone https://github.com/heheda12345/Jenga-SOSP25-AE --branch vllm-v0-baseline
cd vllm-v0-baseline/
pip install -e .
pip uninstall transformers
pip install transformers==4.45.2
```

## vllm-v0-mm-baseline
```bash
cd ~
python3 -m venv venv/vllm-v0-mm-baseline
source venv/vllm-v0-mm-baseline/bin/activate
git clone https://github.com/heheda12345/Jenga-SOSP25-AE --branch vllm-v0-mm-baseline
cd vllm-v0-mm-baseline
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/c83919c7a6bd47bb452321f08017ef5a5cdd553a/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python python_only_dev.py
pip uninstall transformers
pip install transformers==4.45.2
```

## vllm-v0-pyramidinfer
```bash
cd ~
python3 -m venv venv/vllm-v0-pyramidinfer
source venv/vllm-v0-pyramidinfer/bin/activate
git clone git@github.com:heheda12345/vllm-mem.git --branch pyramidinfer-v3 vllm-v0-pyramidinfer
cd vllm-v0-pyramidinfer/
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/717a5f82cda6dd6a52be6504179adaa64bbdc67a/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python python_only_dev.py
pip uninstall transformers
pip install transformers==4.45.2
pip install sortedcontainers
```

## vllm-v0-pyramidinfer-baseline
```bash
cd ~
python3 -m venv venv/vllm-v0-pyramidinfer-baseline
source venv/vllm-v0-pyramidinfer-baseline/bin/activate
git clone git@github.com:heheda12345/vllm-mem.git --branch pyramidinfer-baseline vllm-v0-pyramidinfer-baseline
cd vllm-v0-pyramidinfer-baseline/
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/717a5f82cda6dd6a52be6504179adaa64bbdc67a/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python python_only_dev.py
pip uninstall transformers
pip install transformers==4.45.2
```

## vllm-v1
```bash
cd ~
python3 -m venv ~/venv/vllm-v1
source ~/venv/vllm-v1/bin/activate
git clone https://github.com/heheda12345/Jenga-SOSP25-AE --branch vllm-v1
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/98d01d3ce2a4d06e85348e375e726c40bee0bdf0/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl pip install --editable .
pip uninstall transformers
pip install transformers==4.54.0
pip install datasets
```

