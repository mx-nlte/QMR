// This source code is subject to the GNU General Public License version 3
// and provided WITHOUT ANY WARRANTY.
// Author: Maximilian Nolte at Technische Universtaet Darmstadt

#ifndef EIGEN_QMRWLA_H
#define EIGEN_QMRWLA_H

namespace Eigen {

namespace internal {

/**
* Quasi-Minimal Residual Method with out look ahead Lanczos
* based on the algorithm of Freund and Nachtigal 1991.
* This algorithm is based on the Lanczos algorithm with coupled two term recurrences
+ without look ahead steps.
*
* Parameters:
*  \param mat       matrix of linear system of equations
*  \param rhs       right hand side vector of linear system of equations
*  \param x         on input: initial guess, on output: solution
*  \param precond   preconditioner used
*  \param iters     on input: maximum number of iterations to perform
*                   on output: number of iterations performed
*  \param restart   number of iterations for a restart
*  \param tol_error on input: relative residual tolerance
*                   on output: residuum achieved
*
* \sa IterativeMethods::bicgstab()
*
*
* For references, please see:
*
* Barrett, Richard and Berry, Michael W and Chan, Tony F and Demmel, James and Donato et al.
* Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods
* Society for Industrial and Applied Mathematics, Philadelphia, USA, 64-68 (1994)
*
* Roland W. Freund and Noel M. Nachtigal
* An Implementation of the QMR Method Based on Coupled Two Term Recurrences
* SIAM: Journal on Scientific Computing 15.2, 313-337 (1994)
* 
* Roland W. Freund and Noel M. Nachtigal
* QMR: A Quasi-Minimal Residual Method for Non-Hermitian Linear Systems.
* Numer. Math. 60, 315-339 (1991)
* 
*/
template<typename Dest, typename VectorType>
inline typename Dest::Scalar vecProdTransposed(VectorType &a, VectorType &b){
  return a.transpose() * b;
}

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
bool qmrWLA(const MatrixType & mat, const Rhs & rhs, Dest & x, const Preconditioner & precond,
    Index &iters, const Index &restart, typename Dest::RealScalar & tol_error) {
         
  using std::sqrt;
  using std::abs;

  typedef typename Dest::RealScalar RealScalar;
  typedef typename Dest::Scalar Scalar;
  typedef Matrix < Scalar, Dynamic, 1 > VectorType;
  typedef Matrix < RealScalar, Dynamic, 1 > RealVectorType;
  typedef Matrix < Scalar, Dynamic, Dynamic, ColMajor> FMatrixType;

  // left preconditioning
  // Preconditioner precond1 = precond;
  // Eigen::IdentityPreconditioner precond2 = Eigen::IdentityPreconditioner();

  // right preconditioning
  Eigen::IdentityPreconditioner precond1 = Eigen::IdentityPreconditioner();
  Preconditioner precond2 = precond;


  const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();

  if(rhs.norm() <= considerAsZero) 
  {
    	x.setZero();
    	tol_error = 0;
    	return true;
  }

  RealScalar tol = tol_error;
  const Index maxIters = iters;
  iters = 0;

  VectorType d, p, s, q;
  RealScalar theta;
  Scalar delta, epsilon;

  RealScalar rhsNorm = rhs.norm();
  VectorType residual = rhs - mat * x;
  VectorType v1 = residual;
  VectorType y = precond1.solve(v1);
  VectorType w1 = residual;
  VectorType z = precond2.solve(w1); // transpose Precond2

  RealScalar abs1 = y.norm();
  RealScalar abs2 = z.norm();

  RealScalar gamma = 1;
  Scalar eta = -1;

  for (Index i = 1; i <= maxIters; i++)
  {
      iters++;
      if (abs1 < 1e-13 || abs2 < 1e-13)
      {
  	    std::cout << "QMR failed; abs1 = " << abs1 << " or " << "abs2 = " << abs2 << std::endl;
  	    return false;
  	}

    VectorType v = v1/abs1;
  	VectorType w = w1/abs2;
    y = y/abs1;
    z = z/abs2;

  	delta = vecProdTransposed<VectorType>(z,y);
  	if (abs(delta) < 1e-10) {
  		std::cout << "QMR failed; delta = " << abs(delta) << std::endl;
  		return false;
  	}
    VectorType y1 = precond2.solve(y);
    VectorType z1 = precond1.solve(z); // transpose Precond1

  	if (i == 1)
      {
  		p = y1;
  		q = z1;
  	}
  	else
      {
  		p = y1 - (abs2*delta/epsilon) * p;
  		q = z1 - (abs1*delta/epsilon) * q;
  	}
      
  	VectorType p1 = mat * p;
  	epsilon = vecProdTransposed<VectorType>(q,p1);
  	if (abs(epsilon) < 1e-13)
      {
  		std::cout << "QMR failed; epsilon = " << abs(epsilon) << std::endl;
  		return false;
  	}

  	Scalar beta = epsilon/delta;
  	if (abs(beta) < 1e-13) {
  		std::cout << "QMR failed; beta = " << abs(beta) << std::endl;
  		return false;
  	}

  	v1 = p1 - beta * v;

    y = precond1.solve(v1);

  	RealScalar abs1Old = abs1;
  	abs1 = y.norm();
  	w1 = mat.transpose() * q - beta * w;

    z = precond2.solve(w1); // transpose Precond2
  	abs2 = z.norm();

  	RealScalar thetaOld = theta;
  	theta = abs1/(gamma*abs(beta));
  	RealScalar gammaOld = gamma;
  	gamma = 1/(sqrt(1 + theta*theta));
  	if (abs(theta) < 1e-13) {
  		std::cout << "QMR failed; theta = " << abs(theta) << std::endl;
  		return false;
  	}

  	eta = - eta*abs1Old*gamma*gamma/(beta*gammaOld*gammaOld);

  	if (i == 1) {
  		d = eta * p;
  		s = eta * p1;
  	}
  	else {
  		d = eta * p + (thetaOld*gamma)*(thetaOld*gamma) * d;
  		s = eta * p1 + (thetaOld*gamma)*(thetaOld*gamma) * s;
  	}

  	x += d;
  	residual -= s;

  	RealScalar residualNorm = residual.norm();
  	tol_error = residualNorm/rhsNorm;

  	if (tol_error < tol) {
  		return true;
  	}
  }
	std::cout << "QMR finished with max iterations" << std::endl;
	return false;
}

}

template< typename _MatrixType,
          typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
class QMRWLA;

namespace internal {

template< typename _MatrixType, typename _Preconditioner>
struct traits<QMRWLA<_MatrixType,_Preconditioner> >
{
  typedef _MatrixType MatrixType;
  typedef _Preconditioner Preconditioner;
};

}

/** \ingroup IterativeLinearSolvers_Module
  * \brief A QMR without Look Ahead solver for sparse square problems
  *
  * This class allows to solve for A.x = b sparse linear problems using a quasi minimal
  * residual method. The vectors x and b can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, can be a dense or a sparse matrix.
  * \tparam _Preconditioner the type of the preconditioner. Default is DiagonalPreconditioner
  *
  * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
  * and setTolerance() methods. The defaults are the size of the problem for the maximal number of iterations
  * and NumTraits<Scalar>::epsilon() for the tolerance.
  *
  * This class can be used as the direct solver classes. Here is a typical usage example:
  * \code
  * int n = 10000;
  * VectorXd x(n), b(n);
  * SparseMatrix<double> A(n,n);
  * // fill A and b
  * QMRWLA<SparseMatrix<double> > solver(A);
  * x = solver.solve(b);
  * std::cout << "#iterations:     " << solver.iterations() << std::endl;
  * std::cout << "estimated error: " << solver.error()      << std::endl;
  * // update b, and solve again
  * x = solver.solve(b);
  * \endcode
  *
  * By default the iterations start with x=0 as an initial guess of the solution.
  * One can control the start using the solveWithGuess() method.
  * 
  * QMR without Look Ahead can also be used in a matrix-free context, see the following \link MatrixfreeSolverExample example \endlink.
  *
  * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
  */
template< typename _MatrixType, typename _Preconditioner>
class QMRWLA : public IterativeSolverBase<QMRWLA<_MatrixType,_Preconditioner> >
{
  typedef IterativeSolverBase<QMRWLA> Base;
  using Base::matrix;
  using Base::m_error;
  using Base::m_iterations;
  using Base::m_info;
  using Base::m_isInitialized;

public:
  using Base::_solve_impl;
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix < RealScalar, Dynamic, 1 > RealVectorType;
  typedef _Preconditioner Preconditioner;

private:
  Index m_restart;

public:

  /** Default constructor. */
  QMRWLA() : Base(), m_restart(30) {}

  /** Initialize the solver with matrix \a A for further \c Ax=b solving.
    *
    * This constructor is a shortcut for the default constructor followed
    * by a call to compute().
    *
    * \warning this class stores a reference to the matrix A as well as some
    * precomputed values that depend on it. Therefore, if \a A is changed
    * this class becomes invalid. Call compute() to update it with the new
    * matrix A, or modify a copy of A.
    */
  template<typename MatrixDerived>
  explicit QMRWLA(const EigenBase<MatrixDerived>& A) : Base(A.derived()), m_restart(30) {}

  ~QMRWLA() {}

  /** Get the number of iterations after that a restart is performed.
    */
  Index get_restart() { return m_restart; }

  /** Set the number of iterations after that a restart is performed.
    *  \param restart   number of iterations for a restarti, default is 30.
    */
  void set_restart(const Index restart) { m_restart=restart; }

  /** \internal */
  template<typename Rhs,typename Dest>
  void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const
  {
    m_iterations = Base::maxIterations();
    m_error = Base::m_tolerance;
    bool ret = internal::qmrWLA(matrix(), b, x, Base::m_preconditioner, m_iterations, m_restart, m_error);
    m_info = (!ret) ? NumericalIssue
          : m_error <= Base::m_tolerance ? Success
          : NoConvergence;
  }

protected:

};

} // end namespace Eigen

#endif // EIGEN_QMRWLA_H
