import re
import setuptools


with open("ehseg/__init__.py", encoding="utf-8") as f:
    version = re.search(r"__version__\s*=\s*'(\S+)'", f.read()).group(1)


def parse_requirements_file(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [line.strip() for line in fid.readlines() if line]

    return requires


install_requires = parse_requirements_file('requirements.txt')
# nm_requires = [r.split('>=')[0] for r in install_requires]
# if sys.version_info[0] == 2:
#     install_requires = install_requires
# if sys.version_info[0] == 3:
#     install_requires = nm_requires

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
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ]
)
