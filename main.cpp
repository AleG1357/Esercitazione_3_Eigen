#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include <iomanip>



using namespace std;    // (non scrivo std:: ogni volta)
using namespace Eigen;   //per usare le classi di eigen senza 'Eigen::'

// Definisco i sistemi (A , b)

// Sistema 1
MatrixXd A1() {
    MatrixXd A(2,2);
    A <<  5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    return A;
}

VectorXd b1() {
    VectorXd b(2);
    b << -5.169911863249772e-01,
          1.672384680188350e-01;
    return b;
}

// Sistema 2
MatrixXd A2() {
    MatrixXd A(2,2);
    A << 5.547001962252291e-01, -5.540607316466765e-01,
         8.320502943378437e-01, -8.324762492991313e-01;
    return A;
}

VectorXd b2() {
    VectorXd b(2);
    b << -6.394645785530173e-04,
          4.259549612877223e-04;
    return b;
}

// Sistema 3
MatrixXd A3() {
    MatrixXd A(2,2);
    A << 5.547001962252291e-01, -5.547001955851905e-01,
         8.320502943378437e-01, -8.320502947645361e-01;
    return A;
}

VectorXd b3() {
    VectorXd b(2);
    b << -6.400391328043042e-10,
          4.266924591433963e-10;
    return b;
}


// Calcola la soluzione usando LU (con pivot parziale)
VectorXd solveLU(const MatrixXd& A, const VectorXd& b)
{
    PartialPivLU<MatrixXd> lu(A); 
    // oppure: FullPivLU<MatrixXd> lu(A);
    // ma  PartialPivLU dovrebbe bastare per un sistema2x2.
    return lu.solve(b);
}

// Calcola la soluzione usando la decomposizione QR (Householder)
VectorXd solveQR(const MatrixXd& A, const VectorXd& b)
{
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

// Calcola l'errore relativo rispetto ad un vettore xExact
double relativeError(const VectorXd& xNum, const VectorXd& xExact)
{
    return (xNum - xExact).norm() / xExact.norm();
}

int main()
{
    // Soluzione esatta di tutti i sistemi 
    VectorXd xExact(2);
    xExact << -1.0, -1.0;

    // --------- Sistema 1 -----------
    cout << "\n=== SISTEMA 1 ===" << endl;
    MatrixXd A = A1();
    VectorXd b = b1();
    cout << "A =\n" << A << endl;
    cout << "b =\n" << b.transpose() << endl;

    VectorXd xLu = solveLU(A, b);
    VectorXd xQr = solveQR(A, b);
    double errLu = relativeError(xLu, xExact);
    double errQr = relativeError(xQr, xExact);

	cout << std::setprecision(15);   //Voglio evidenziare che le componenti della soluzione non sono esattamente 1

    cout << "Soluzione LU  : " << xLu.transpose() << endl;
    cout << "Soluzione QR  : " << xQr.transpose() << endl;
	
	cout << std::setprecision(6);  //Torno a 6 cirfre per avere un output più leggibile
	
    cout << "Err. rel. LU  : " << errLu << endl;
    cout << "Err. rel. QR  : " << errQr << endl;

    // --------- Sistema 2 -----------
    cout << "\n=== SISTEMA 2 ===" << endl;
    A = A2();
    b = b2();
    cout << "A =\n" << A << endl;
    cout << "b =\n" << b.transpose() << endl;

    xLu = solveLU(A, b);
    xQr = solveQR(A, b);
    errLu = relativeError(xLu, xExact);
    errQr = relativeError(xQr, xExact);
	
	

	cout << std::setprecision(15);

    cout << "Soluzione LU  : " << xLu.transpose() << endl;
    cout << "Soluzione QR  : " << xQr.transpose() << endl;
	
	cout << std::setprecision(6); 
	
    cout << "Err. rel. LU  : " << errLu << endl;
    cout << "Err. rel. QR  : " << errQr << endl;

    // --------- Sistema 3 -----------
    cout << "\n=== SISTEMA 3 ===" << endl;
    A = A3();
    b = b3();
    cout << "A =\n" << A << endl;
    cout << "b =\n" << b.transpose() << endl;

    xLu = solveLU(A, b);
    xQr = solveQR(A, b);
    errLu = relativeError(xLu, xExact);
    errQr = relativeError(xQr, xExact);
	
	cout << std::setprecision(15);

    cout << "Soluzione LU  : " << xLu.transpose() << endl;
    cout << "Soluzione QR  : " << xQr.transpose() << endl;
	
	cout << std::setprecision(6);
	
    cout << "Err. rel. LU  : " << errLu << endl;
    cout << "Err. rel. QR  : " << errQr << endl;

    return 0;
}

