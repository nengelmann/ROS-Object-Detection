from setuptools import setup

package_name = 'objectdetection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['./model/coco.names']),
        ('share/' + package_name, ['./model/yolov3.cfg']),
        ('share/' + package_name, ['./model/yolov3.weights']),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='youremail@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = objectdetection.publisher_member_function:main',
            'listener = objectdetection.subscriber_member_function:main',
            'node = objectdetection.node:main',
        ],
    },
)