from pathlib import Path

from setuptools import find_packages, setup

PY_TXI_VERSION = "0.10.0"

common_setup_kwargs = {
    "author": "Ilyas Moutawwakil",
    "author_email": "ilyas.moutawwakil@gmail.com",
    "description": "A Python wrapper around TGI and TEI servers",
    "keywords": ["tgi", "llm", "tei", "embedding", "huggingface", "docker", "python"],
    "url": "https://github.com/IlyasMoutawwakil/py-txi",
    "long_description_content_type": "text/markdown",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "platforms": ["linux", "windows", "macos"],
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
    ],
}


setup(
    name="py-txi",
    version=PY_TXI_VERSION,
    packages=find_packages(),
    install_requires=["docker", "huggingface-hub", "numpy", "aiohttp"],
    extras_require={"quality": ["ruff"], "testing": ["pytest"]},
    **common_setup_kwargs,
)
