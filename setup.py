import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="videoqa_eval",  # 模块名称
    version="1.0",  # 当前版本
    author="Sicheng Z",  # 作者
    author_email="Kyuushi94@outlook.com",  # 作者邮箱
    description="A comprehensive metric toolset for evaluation NLG and videoQA tasks.",  # 模块简介
    long_description=long_description,  # 模块详细介绍
    long_description_content_type="text/markdown",  # 模块详细介绍格式
    url="https://github.com/fesvhtr/VideoQA-Eval",  # 模块github地址
    packages=setuptools.find_packages(),  # 自动找到项目中导入的模块
    # 模块相关的元数据
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # 依赖模块
    install_requires=[
        'pillow',
    ],
    python_requires='>=3',
)