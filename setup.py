import setuptools

def read(path):
    with open(path, encoding = 'utf-8') as f:
        return f.read()


setuptools.setup(
    name="rlig",
    version="0.0.0",
    author="Jordan Taylor",
    author_email="jt2006@bath.ac.uk",
    description="An implementation of reinforcement learning with imagined goals",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/jordan-wei-taylor/imagined-goals",
    project_urls={
        "Bug Tracker": "https://github.com/jordan-wei-taylor/imagined-goals/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    license=read('LICENSE'), 
    install_requires=[

    ]
)