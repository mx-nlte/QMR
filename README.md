# QMR

This repository contains the C++ header files implemented within my bachelor thesis at TU Darmstadt, Department of Electrical Engineering and Information Technology.

The QMR solver can be used in the same way as iterative solver from Eigen. Just include the header files and define the parameters. For the numerical examples carried
out with the Bembel framework for boundary element methods only dense assembled matrices were investigated.
For this examples load the unofficial fork from https://github.com/flx-wlf/bembel/tree/dense_matrix to use the routine which assembles dense system matrices, e.g.

Eigen::MatrixXd myDiscDense = denseAssembly(myDisc);