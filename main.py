import time
import matplotlib.pyplot as plt
from operacje import *

#przykladowe parametry
a2 = -1
a3 = -1
N = 973
f = 8
#wektor pobudzenia b
b = []
#macierz systemowa A
A = []


# utworzenie przykladowego wektora pobudzenia b
def stworz_b(n):
    b = [[0] for _ in range(n)]

    for i in range (n):
        b[i][0] = math.sin(i *(f + 1))

    return b

# utworzenie przykladowej macierzy systemowej A
def stworz_A(a1, n):
    A = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = a1
            if (i - 1) == j or (j - 1) == i:
                A[i][j] = a2
            if (i - 2) == j or (j - 2) == i:
                A[i][j] = a3

    return A

# rozwiazanie ukladu rownan metoda Jacobiego
def B_metoda_iteracyjna_Jacobiego(n):
    # Ax = b
    # pomocnicze kopie:
    start1 = time.perf_counter()
    X_prev = jedynki_wektor(n)
    X = jedynki_wektor(n)
    normy_bledu_rezydualnego = []
    blad = 1/1000000000
    norm_bl = 1
    liczba_iteracji = 0
    while norm_bl > blad and liczba_iteracji<30:
        for i in range(n):
            # suma1:
            sum1 = 0
            for j in range(0, i):
                sum1 += A[i][j] * X_prev[j][0]
            # suma2:
            sum2 = 0
            for j in range(i+1, n):
                sum2 += A[i][j] * X_prev[j][0]

            X[i][0] = (b[i][0] - sum1 - sum2)/A[i][i]

        wektor_residuum = odejmowanie_macierzy(mnozenie_macierzy(A, X), b)
        norm_bl = norma_residuum(wektor_residuum)
        normy_bledu_rezydualnego.append(norm_bl)
        X_prev = przypisz_macierz_pomocnicza(X)
        liczba_iteracji += 1

    koniec1 = time.perf_counter()
    czas = koniec1 - start1
    print("CZAS: " + str(czas))

    #for i in range(n):
    #    print(X[i])
    print("Liczba iteracji: " + str(liczba_iteracji))

    if(n == 973):
        # wykres normy bledu rezydualnego
        iteracje = []
        for i in range(1, liczba_iteracji + 1):
            iteracje.append(i)

        plt.plot(iteracje, normy_bledu_rezydualnego, color='hotpink')
        plt.yscale('log')
        plt.title("normy bledu rezydualnego dla kolejnych liczby iteracji - metoda Jacobiego")
        plt.xlabel('numer iteracji')
        plt.ylabel('norma bledu rezydualnego')
        plt.grid(True)
        plt.savefig("result/norma_bledu_Jacobi" + str(A[0][0]))
        plt.show()
        plt.figure()

    return liczba_iteracji

# rozwiazanie ukladu rownan metoda Gaussa Seidela
def  B_metoda_iteracyjna_Gaussa_Seidela(n):
    # Ax = b
    # pomocnicze kopie:
    start1 = time.perf_counter()
    X_prev = jedynki_wektor(n)
    X = jedynki_wektor(n)
    normy_bledu_rezydualnego = []
    blad = 1 / 1000000000
    norm_bl = 1
    liczba_iteracji = 0
    while (norm_bl > blad and liczba_iteracji<30):
        # dla a1 = 3 by wygenerowac wykres ktory pokaze ze norma bledu rosnie potrzebny jest dodatkowy warunek -
        # koniec gdy liczba iteracji jest mniejsza od danej liczby zeby algorytm mogl sie skonczyc -
        # nie zbiega sie rozw dlatego norma caly czas rosnie
        for i in range(n):
            # suma1:
            sum1 = 0
            for j in range(0, i):
                sum1 += A[i][j] * X[j][0]
            # suma2:
            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i][j] * X_prev[j][0]

            X[i][0] = (b[i][0] - sum1 - sum2) / A[i][i]

        wektor_residuum = odejmowanie_macierzy(mnozenie_macierzy(A, X), b)
        norm_bl = norma_residuum(wektor_residuum)
        normy_bledu_rezydualnego.append(norm_bl)
        X_prev = przypisz_macierz_pomocnicza(X)
        liczba_iteracji += 1

    koniec1 = time.perf_counter()
    czas = koniec1 - start1
    print("CZAS: " + str(czas))

    #for i in range(n):
    #    print(X[i])
    print("Liczba iteracji: " + str(liczba_iteracji))

    if n == 973:
        # wykres norny bledu rezydualnego
        iteracje = []
        for i in range(1, liczba_iteracji + 1):
            iteracje.append(i)

        plt.plot(iteracje, normy_bledu_rezydualnego, color='cyan')
        plt.yscale('log')
        plt.title("normy bledu rezydualnego dla kolejnych liczby iteracji - metoda Gaussa")
        plt.xlabel('numer iteracji')
        plt.ylabel('norma bledu rezydualnego')
        plt.grid(True)
        plt.savefig("result/norma_bledu_Gauss" + str(A[0][0]))
        plt.show()
        plt.figure()

    return liczba_iteracji

# rozwiazanie ukladu rownan metoda faktoryzacji LU
def metoda_faktoryzacji_LU():
    # Ax = b
    # LUx = b
    # Ly = b,        Uy = z
    L, U = macierze_trojkatne_LU(A)
    b_pom = przypisz_macierz_pomocnicza(b)
    y = zera_macierz(len(A[0]), len(b[0]))

    for i in range(len(L)):
        sum = 0
        for j in range(0, i):
            sum +=  L[i][j] * y[j][0]
        y[i][0] = b_pom[i][0] - sum
        y[i][0] /= L[i][i]

    x = zera_macierz(len(A[0]), len(b[0]))
    for i in range(len(U)-1, -1, -1):
        sum = 0
        for j in range(len(U)-1, i, -1):
            sum += U[i][j] * x[j][0]
        x[i][0] = y[i][0] - sum
        x[i][0] /= U[i][i]

    if(len(A[0]) == 973):
        wektor_residuum = odejmowanie_macierzy(mnozenie_macierzy(A, x), b)
        norm_bl = norma_residuum(wektor_residuum)
        print("Norma bledu rezydualnego dla N=973, a1=3, faktoryzacja LU: r=" + str(norm_bl))

    #for i in range(len(x)):
    #    print(x[i])


# badanie metod interacyjnych:
# a1 = 13
A = stworz_A(13, N)
b = stworz_b(N)
print("WYNIK dla metody Jacobiego, a1 = 13")
B_metoda_iteracyjna_Jacobiego(N)
print("WYNIK dla metody Gaussa_Seidela, a1 = 13")
B_metoda_iteracyjna_Gaussa_Seidela(N)

# a1 = 3
A = stworz_A(3, N)
b = stworz_b(N)
print("WYNIK dla metody Jacobiego, a1 = 3")
B_metoda_iteracyjna_Jacobiego(N)
print("WYNIK dla metody Gaussa_Seidela, a1 = 3")
B_metoda_iteracyjna_Gaussa_Seidela(N)

# badanie metody faktoryzacji LU
A = stworz_A(3, N)
b = stworz_b(N)
print("WYNIK dla bezposredniej metody - faktoryzacja LU, a1 = 3")
metoda_faktoryzacji_LU()

# badanie czasow
N = [100, 200, 500, 1000, 2000, 3000]
# 1) metoda Jacobiego
czas_Jacobi = []
iteracje_Jacobi = []
for i in N:
    A = stworz_A(13, i)
    b = stworz_b(i)
    t1 = time.perf_counter()
    iteracje_Jacobi.append(B_metoda_iteracyjna_Jacobiego(i))
    t2 = time.perf_counter()
    czas1 = t2 - t1
    czas_Jacobi.append(czas1)

# 2) metoda Gaussa Seidela
czas_Gauss = []
iteracje_Gauss = []
for i in N:
    A = stworz_A(13, i)
    b = stworz_b(i)
    t1 = time.perf_counter()
    iteracje_Gauss.append(B_metoda_iteracyjna_Gaussa_Seidela(i))
    t2 = time.perf_counter()
    czas2 = t2 - t1
    czas_Gauss.append(czas2)

# 3) metoda faktoryzacji LU
czas_LU = []
for i in N:
    A = stworz_A(13, i)
    b = stworz_b(i)
    t1 = time.perf_counter()
    metoda_faktoryzacji_LU()
    t2 = time.perf_counter()
    czas3 = t2 - t1
    czas_LU.append(czas3)


print("czas Jacobi:")
for i in czas_Jacobi:
    print(i)
print("\nczas Gauss-Seidel:")
for i in czas_Gauss:
    print(i)
print("\nczas LU")
for i in czas_LU:
    print(i)

# wizualizacja wyresami
plt.plot(N, czas_Jacobi, color='hotpink', label='metoda Jacobiego')
plt.plot(N, czas_Gauss, color='cyan', label='metoda Gaussa Seidela')
plt.plot(N, czas_LU, color='blue', label='faktoryzacja LU')
plt.title("czas wykonywania metody w zaleznosci od N")
plt.legend()

plt.xlabel('N')
plt.ylabel('czas[s]')
plt.grid(True)
plt.savefig("result/czasyWykonania")
plt.show()
plt.figure()

plt.plot(N, iteracje_Jacobi, color='hotpink', label='metoda Jacobiego')
plt.plot(N, iteracje_Gauss, color='cyan', label='metoda Gaussa Seidela')
plt.title("liczba iteracji w zaleznosci od N")
plt.legend()
plt.xlabel('N')
plt.ylabel('liczba iteracji')
plt.grid(True)
plt.savefig("result/iteracje")
plt.show()
plt.figure()
