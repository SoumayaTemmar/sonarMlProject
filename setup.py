from setuptools import setup, find_packages

def get_requirements(file_path):
   requirements = []

   with open(file_path, 'r') as file_obj:
      requirements = file_obj.readlines()
      requirements = [req.replace('\n', '') for req in requirements]

      if '-e .' in requirements:
         requirements.remove('-e .')


setup(
   name='sonarMlProject',
   version='0.1.0',
   author='soumaya',
   author_email='temmars32@gmail.com',
   packages=find_packages(),
   install_requires= get_requirements('requirements.txt')
)
