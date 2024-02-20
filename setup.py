from pathlib import Path

from setuptools import find_packages, setup

PY_TGI_VERSION = "0.1.3"

common_setup_kwargs = {
    "author": "Ilyas Moutawwakil",
    "author_email": "ilyas.moutawwakil@gmail.com",
    "description": "A Python wrapper around TGI",
    "keywords": ["python", "tgi", "llm", "huggingface", "docker"],
    "url": "https://github.com/IlyasMoutawwakil/py-tgi",
    "long_description_content_type": "text/markdown",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "platforms": ["linux", "windows", "macos"],
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
    ],
}


setup(
    name="py-tgi",
    version=PY_TGI_VERSION,
    packages=find_packages(),
    install_requires=["docker", "huggingface-hub"],
    extras_require={"quality": ["ruff"]},
    **common_setup_kwargs,
)
