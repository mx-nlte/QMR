// This source code is subject to the GNU General Public License version 3
// and provided WITHOUT ANY WARRANTY.
// Author: Maximilian Nolte at Technische Universtaet Darmstadt

#ifndef EIGEN_QMR_H
#define EIGEN_QMR_H

namespace Eigen {

namespace internal {

/**
* Quasi-Minimal Residual Method
* based on the algorithm of Freund and Nachtigal 1991.
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
* Roland W. Freund and Noel M. Nachtigal
* QMR: A Quasi-Minimal Residual Method for Non-Hermitian Linear Systems.
* Numer. Math. 60, 315-339 (1991)
*
*/

template<typename MatrixType>
inline MatrixType matProd(MatrixType &A, MatrixType &B){
	return A.adjoint() * B;
}

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
bool qmr(const MatrixType & mat, const Rhs & rhs, Dest & x, const Preconditioner & precond,
    Index &iters, const Index &restart, typename Dest::RealScalar & tol_error) {
  
	using std::sqrt;
	using std::abs;

	typedef typename Dest::RealScalar RealScalar;
	typedef typename Dest::Scalar Scalar;
	typedef std::complex<RealScalar> ComplexScalar;
	typedef Matrix < Scalar, Dynamic, 1 > VectorType;
	typedef Matrix < Scalar, 1, Dynamic > RowVectorType;
	typedef Matrix < RealScalar, Dynamic, 1 > RealVectorType;
	typedef Matrix < Scalar, Dynamic, Dynamic, ColMajor> FMatrixType;
	typedef Matrix < ComplexScalar, Dynamic, 1 > EigenvalueType;

	// left preconditioning - bug with residuum
	// Preconditioner precond1 = precond;
	// Eigen::IdentityPreconditioner precond2 = Eigen::IdentityPreconditioner();

	// right preconditioning
	Eigen::IdentityPreconditioner precond1 = Eigen::IdentityPreconditioner();
	Preconditioner precond2 = precond;

	RealScalar tol = tol_error;
	const Index maxIters = iters;
	iters = 0;

	// storage for QR decomposition
	VectorType cAlpha;
	VectorType cBeta;
	VectorType sAlpha;
	VectorType sBeta;

	// Initialisation
	VectorType residual = rhs - mat*x;

	const RealScalar rhsNorm = rhs.norm();
	Scalar rotationFactor = 1;

	VectorType v = precond1.solve(residual);
	residual = v;
	const RealScalar residual0Norm = v.norm();
	v = v/residual0Norm;

	VectorType w = precond2.solve(residual);
	w = w/w.norm();

	// number of blocks;
	int k = 0;
	FMatrixType Vk = v;
	FMatrixType Wk = w;

	FMatrixType Dk = Wk.adjoint() * Vk;
		
	// set zero for the first iteration
	FMatrixType VOld = FMatrixType::Zero(mat.rows(), 1);
	FMatrixType WOld = FMatrixType::Zero(mat.rows(), 1);
	// need to be one for valid inversion in first iteration
	FMatrixType DOld = FMatrixType::Ones(1, 1);

	// VectorType tauTest;
	Scalar tau = residual0Norm;
	FMatrixType Pk = FMatrixType::Zero(mat.rows(),1);
	FMatrixType PkOld = FMatrixType::Zero(mat.rows(),1);
	FMatrixType PkOld2 = FMatrixType::Zero(mat.rows(),1);
	FMatrixType Delta;
	FMatrixType Epsilon;
	FMatrixType Theta;
	int sizeTheta = 0;

	// is initial guess already good enough?
	if(residual0Norm == 0)
	{
		tol_error = 0;
		return true;
	}

	for (Index i = 1; i <= maxIters; i++)
	{
		++iters;
		// Lanczos iteration
		VectorType v1 = precond2.solve(v);
		VectorType w1 = precond1.solve(w);
		VectorType y1 = mat * v1;
		VectorType z1 = mat.adjoint() * w1;
		VectorType y = precond1.solve(y1);
		VectorType z = precond2.solve(z1);

		VectorType alpha = Dk.inverse() * Wk.adjoint() * y;
		VectorType beta = DOld.inverse() * WOld.adjoint() * y;
		VectorType alphaW = Dk.adjoint().inverse() * Vk.adjoint() * z;
		VectorType betaW = DOld.adjoint().inverse() * VOld.adjoint() * z;
		
		// Check singular values of D
		Eigen::ComplexEigenSolver<FMatrixType> ces(Dk, false);
		EigenvalueType eigenvalues = ces.eigenvalues(); 
		RealVectorType eigenvaluesAbs = RealVectorType::Zero(eigenvalues.size());
		for (Index j = 0; j < eigenvalues.size();j++) {
			eigenvaluesAbs(j) = abs(eigenvalues(j));
		}

		RealScalar singularValue = eigenvaluesAbs.minCoeff();

		bool regular = false;
		if (singularValue > 1e-7)
		{
			regular = true;
			k++;
		}

		Delta.conservativeResize(alpha.rows(),alpha.rows());
		Delta.bottomRows(1).setZero(); // Fill last row
		Epsilon.conservativeResize(beta.rows(),Vk.cols());
		Theta.conservativeResize(sizeTheta,Vk.cols());

		if (regular)
		{
			// Regular step
			v = y - Vk * alpha - VOld * beta;
			w = z - Wk * alphaW - WOld * betaW;
		}
		else
		{
			// Inner Step
			alpha = VectorType::Zero(Vk.cols());
			alphaW = VectorType::Zero(VOld.cols());
			v = y - VOld * beta;
			w = z - WOld * betaW;

		}	

		RealScalar abs1 = v.norm();
		RealScalar abs2 = w.norm();

		// At first the last column of He is assembled and after the Givens Rotations it is the last column of R
		VectorType lastColR  = VectorType::Zero(i + 1);

		// Calculate Rotation Matrix
		VectorType c, s;
		c.resize(cBeta.size() + cAlpha.size());
		s.resize(sBeta.size() + sAlpha.size());

		// for the first block were beta does not exist.
		if (k<2)
		{
			lastColR.head(alpha.size()) = alpha;

			c << cAlpha;
			s << sAlpha;
		}
		else
		{
			lastColR.segment(i-alpha.size()-beta.size(),beta.size()) = beta;
			lastColR.segment(i-alpha.size(),alpha.size()) = alpha;

			c << cBeta, cAlpha;
			s << sBeta, sAlpha;
		}
		lastColR(lastColR.size() - 1) = abs1;

		// Givens Rotation

		Eigen::JacobiRotation<Scalar> G;

		const Index ng = c.size();
		for (Index n = 0; n < ng; n++)
		{
			const Index blockInd = i - ng + n - 1;

			G.makeGivens(c(n),s(n));
			lastColR.applyOnTheLeft(blockInd, blockInd + 1, G.transpose());
		}

		// correct last two entries and set next givens parameter c,s
		Scalar mu = lastColR(lastColR.size()-2);
		Scalar nu = lastColR(lastColR.size()-1);
		Scalar cn, sn;
		if (abs(mu) < 1e-7)
		{
			cn = 0;
			sn = 1;
		}
		else
		{
			cn = abs(mu)/sqrt(abs(mu)*abs(mu) + abs(nu) * abs(nu));
			sn = numext::conj(cn * nu/mu);
		}
		Scalar mun = cn*mu + sn*nu;
		lastColR(lastColR.size() - 2) = mun;

		// Reuse former Rotation for next steps
		if (regular)
		{
			cBeta.resize(cAlpha.size());
			sBeta.resize(sAlpha.size());
			cBeta << cAlpha;
			sBeta << sAlpha;
			cAlpha.resize(1);
			cAlpha(0) = cn;
			sAlpha.resize(1);
			sAlpha(0) = sn;
		}
		else
		{
			cAlpha.conservativeResize(cAlpha.size() + 1);
			cAlpha(cAlpha.size()-1) = cn;
			sAlpha.conservativeResize(sAlpha.size() + 1);
			sAlpha(sAlpha.size()-1) = sn;
		}

		// cases for upper triangular block matrix
		if (k < 2)
		{
			Delta.rightCols(1) = lastColR.head(alpha.size());
			Epsilon = FMatrixType::Zero(1,1);
			Theta = FMatrixType::Zero(1,1);
		}
		else if (k < 3)
		{
			Epsilon.rightCols(1) = lastColR.segment(i-alpha.size()-beta.size(),beta.size());
			Delta.rightCols(1) = lastColR.segment(i-alpha.size(),alpha.size());
			Theta = FMatrixType::Zero(1,1);

		}
		else 
		{
			Theta.rightCols(1) = lastColR.segment(i-alpha.size()-beta.size()-sizeTheta,sizeTheta);;
			Epsilon.rightCols(1) = lastColR.segment(i-alpha.size()-beta.size(),beta.size());
			Delta.rightCols(1) = lastColR.segment(i-alpha.size(),alpha.size());
		}
		
		// Update Pk
		Pk = (Vk - PkOld * Epsilon - PkOld2 * Theta) * Delta.inverse();

		if (regular)
		{
			PkOld2 = PkOld;
			PkOld = Pk;
		}
		VectorType dy = Pk.rightCols(1) * (tau * cn);
		x += precond2.solve(dy);
		
		// tau for next iteration after Givens rotation
		tau = - numext::conj(sn) * tau;

		// update upper bound for residualNorm
		rotationFactor *= sn; 
		RealScalar upperBound = residual0Norm * sqrt(i+1) * abs(rotationFactor);
		RealScalar rel_errorBound = upperBound/rhsNorm;

		// Check convergence
		bool checkConvergence = rel_errorBound < tol;

		// Check if relative residual is actually below upper bound
		if (checkConvergence)
		{
			VectorType residualCurrent = rhs - mat * x;
			RealScalar residualCurrentNorm = residualCurrent.norm();
			RealScalar rel_errorCurrent = residualCurrentNorm/rhsNorm;

			if (rel_errorCurrent < tol)
			{
				tol_error = rel_errorCurrent;
				return true;
			}
		}

		// Check for breakdown
		if (abs1 < 1e-7 || abs2 < 1e-7)
		{
			std::cout << "QMR failed; abs1 = " << abs1 << " or " << "abs2 = " << abs2 << std::endl;
			return false;
		}

		// normalise new vectors
		v = v/abs1;
		w = w/abs2;

		// update residual
		residual = abs(sn)*abs(sn) * residual + (cn*tau) * v;
		RealScalar residualNorm = residual.norm();

		assert(residualNorm < upperBound);
		tol_error = residualNorm/rhsNorm;

		// 2nd convergence check with residual
		if (tol_error < tol)
		{
			return true;
		}

		// append new vectors to Vk and Wk
		if (regular)
		{
			// after 2 blocks Theta in matrix R occurs
			if (k > 1) {
				sizeTheta = VOld.cols();
			}

			// Save the old blocks
			VOld = Vk;
			WOld = Wk;
			DOld = Dk;

			// start a new Block
			Vk.resize(mat.rows(),1);
			Wk.resize(mat.rows(),1);

			Vk = v;
			Wk = w;
			Dk = matProd<FMatrixType>(Wk,Vk);
		}
		else
		{
			// Add new vectors to current block
			Vk.conservativeResize(Vk.rows(), Vk.cols()+1);
			Vk.col(Vk.cols()-1) = v;
			Wk.conservativeResize(Wk.rows(), Wk.cols()+1);
			Wk.col(Wk.cols()-1) = w;

			Dk = matProd<FMatrixType>(Wk,Vk);
		}
		
	}
	std::cout << "QMR finished with maximum iterations!" << std::endl;
	return false;
}

}

template< typename _MatrixType,
          typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
class QMR;

namespace internal {

template< typename _MatrixType, typename _Preconditioner>
struct traits<QMR<_MatrixType,_Preconditioner> >
{
  typedef _MatrixType MatrixType;
  typedef _Preconditioner Preconditioner;
};

}

/** \ingroup IterativeLinearSolvers_Module
  * \brief A QMR solver for sparse square problems
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
  * QMR<SparseMatrix<double> > solver(A);
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
  * QMR can also be used in a matrix-free context, see the following \link MatrixfreeSolverExample example \endlink.
  *
  * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
  */
template< typename _MatrixType, typename _Preconditioner>
class QMR : public IterativeSolverBase<QMR<_MatrixType,_Preconditioner> >
{
  typedef IterativeSolverBase<QMR> Base;
  using Base::matrix;
  using Base::m_error;
  using Base::m_iterations;
  using Base::m_info;
  using Base::m_isInitialized;

private:
  Index m_restart;

public:
  using Base::_solve_impl;
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef _Preconditioner Preconditioner;

public:

  /** Default constructor. */
  QMR() : Base(), m_restart(30) {}

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
  explicit QMR(const EigenBase<MatrixDerived>& A) : Base(A.derived()), m_restart(30) {}

  ~QMR() {}

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
    bool ret = internal::qmr(matrix(), b, x, Base::m_preconditioner, m_iterations, m_restart, m_error);
    m_info = (!ret) ? NumericalIssue
          : m_error <= Base::m_tolerance ? Success
          : NoConvergence;
  }

protected:

};

} // end namespace Eigen

#endif // EIGEN_QMR_H
