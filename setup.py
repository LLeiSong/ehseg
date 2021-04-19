import re
import setuptools


with open("ehseg/__init__.py", encoding="utf-8") as f:
    version = re.search(r"__version__\s*=\s*'(\S+)'", f.read()).group(1)

setuptools.setup(
    name="ehseg",
    version=version,
    url="https://github.com/LLeiSong/ehseg",
    author="Lei Song",
    author_email="lsong@clarku.edu",
    description="Edge-highlight Image Segmentation",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    keywords="segmentation, remote sensing",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ]
)
