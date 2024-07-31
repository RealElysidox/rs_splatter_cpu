# 🌟 Simple Gaussian Splatting

## 🔍 About

This repo is a simple reimplementation of the framework for rendering and modelling part of Gaussian Splatting. It only preserves the core functionality of the algorithm but enhances the readability and maintainbility of the code. The only heavy dependencies are opencv and matplotlib, and simulate the rendering process on CPU rather than CUDA, which is convenient for reproducing and code transfering. 

## 📚 Usage

### Installation

```bash
## clone the repo
git clone https://github.com/RealElysidox/simple_gaussian.git

## install the dependencies
pip install -r requirement.txt

## splat!
python main.py
```

And you will get a simple illustration:
<img src="asset\3dgs.png">