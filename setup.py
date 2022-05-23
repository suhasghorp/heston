import distutils.core
import Cython.Build

distutils.core.setup(
    ext_modules=Cython.Build.cythonize("HestonPutCombined.pyx", compiler_directives={"language_level": "3"})
)