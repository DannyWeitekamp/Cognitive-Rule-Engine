from setuptools import setup, find_packages 
  
with open('requirements.txt') as f: 
    requirements = f.readlines() 
  
long_description = '...need to add description' 
  
setup( 
        name ='cre', 
        version ='0.1.0', 
        author ='Daniel Weitekamp', 
        author_email ='weitekamp@cmu.edu', 
        url ='https://github.com/DannyWeitekamp/Cognitive-Rule-Engine', 
        description ='A rule engine for Python powered by numba.', 
        long_description = long_description, 
        long_description_content_type ="text/markdown", 
        license ='MIT', 
        packages = find_packages(), 
        # scripts=['bin/altrain'],
        # entry_points ={ 
            
        # }, 
        entry_points={
            "console_scripts": [
                "cre = cre.command_line:main"
            ]
        },

        classifiers =( 
            "Programming Language :: Python :: 3", 
            "License :: OSI Approved :: MIT License", 
            "Operating System :: OS Independent", 
        ), 
        keywords ='expert system production rules', 
        install_requires = requirements, 
) 
