import imagej
import os

os.environ['JAVA_HOME']='C:/ProgramData/Oracle/Java/javapath/java.exe'

ij = imagej.init()
print(ij.getVersion())