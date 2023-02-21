from setuptools import setup, find_packages, Extension
import os, sys, sysconfig
import re

# Try to get the version from 
VERSIONFILE="cre/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# Read requirements.txt for requirements
with open('requirements.txt') as f: 
    requirements = f.readlines() 

  
long_description = '...need to add description' 

dev_requirements = [
    "pytest",
    "pytest-benchmark"
]


def ensure_numba():
    from importlib import import_module
    print("Start")
    numba = None
    # while(numba is None):
    try:
        print("try")
        numba = import_module("numba")
    except ModuleNotFoundError:
        print("NO Numba")
        try:
            import pip
            for numba_req in requirements:
                if("numba" in numba_req):
                    break
            print("numba_req", numba_req)
            if("numba" not in numba_req):
                return
            thing = pip.main(['install', numba_req])  
        except:
            pass
    try:
        numba = import_module("numba")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Failed to install module 'numba' automatically.")
    return numba


def get_ext_modules():
    # Check that python development headers are installed
    python_headers_path = sysconfig.get_config_vars()['INCLUDEPY']
    if not os.path.exists(python_headers_path):
        major = sys.version_info.major
        minor = sys.version_info.minor
        raise ModuleNotFoundError(
    f"Ensure python headers installed: `sudo apt-get install libpython{major}.{minor}-dev`"
        )

    numba = ensure_numba()
    numba_path = numba.extending.include_path()
    cre_c_funcs = Extension(
        name='cre_cfuncs', 
        sources=['cre/cfuncs/cre_cfuncs.c'],
        include_dirs=[numba_path, sysconfig.get_path('include')]
    )
    return [cre_c_funcs]

  
setup( 
        name ='cre', 
        version = __version__, 
        author ='Daniel Weitekamp', 
        author_email ='weitekamp@cmu.edu', 
        url ='https://github.com/DannyWeitekamp/Cognitive-Rule-Engine', 
        description ='A rule engine for Python powered by numba.', 
        long_description = long_description, 
        long_description_content_type ="text/markdown", 
        license ='MIT', 
        packages = find_packages(include=['cre','cre_caching']), 
        
        entry_points={
            "console_scripts": [
                "cre = console.cre_exec:main"
            ]
        },
        ext_modules = get_ext_modules(),


        classifiers =[ 
            "Programming Language :: Python :: 3", 
            "License :: OSI Approved :: MIT License", 
            "Operating System :: OS Independent", 
        ], 
        keywords ='expert system production rules', 
        install_requires = requirements, 
        setup_requires = ['numba'],
        extras_require={
            'dev' : dev_requirements
        }
) 
