from setuptools import setup, find_packages, Extension 

  
with open('requirements.txt') as f: 
    requirements = f.readlines() 

  
long_description = '...need to add description' 

dev_requirements = [
    "pytest",
    "pytest-benchmark"
]

def get_ext_modules():
    import numba
    numba_path = numba.extending.include_path()
    cre_c_funcs = Extension(
        name='cre_cfuncs', 
        sources=['cre/cfuncs/cre_cfuncs.c'],
        include_dirs=[numba_path]
    )
    return [cre_c_funcs]

  
setup( 
        name ='cre', 
        version ='0.3.0', 
        author ='Daniel Weitekamp', 
        author_email ='weitekamp@cmu.edu', 
        url ='https://github.com/DannyWeitekamp/Cognitive-Rule-Engine', 
        description ='A rule engine for Python powered by numba.', 
        long_description = long_description, 
        long_description_content_type ="text/markdown", 
        license ='MIT', 
        packages = find_packages(include=['cre','cre_caching']), 
        # scripts=['bin/altrain'],
        # entry_points ={ 
            
        # }, 
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
