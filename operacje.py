import math

# generowanie wektora z wartosciami 1 o dlugosci N
def jedynki_wektor(N):
    j = [[1] for _ in range(N)]
    return j

#generowanie macierzy m x n wypelnionej zerami
def zera_macierz(n, m):
     z = [[0 for _ in range(m)] for _ in range(n)]
     return z

# implementacja mnozenia dwoch macierzy
def mnozenie_macierzy(A, B):
    C = zera_macierz(len(A), len(B[0]))
    if len(A[0]) == len(B):
        for w in range(len(A)):
            for k in range(len(B[0])):
                for m in range(len(B)):
                    C[w][k] += A[w][m] * B[m][k]
    return C

#implementacja odejmowania dwoch macierzy
def odejmowanie_macierzy(A, B):
    C = zera_macierz(len(A), len(A[0]))
    if(len(A) == len(B) and len(A[0]) == len(B[0])):
        for i in range (len(A)):
            for j in range (len(A[0])):
                C[i][j] = A[i][j] - B[i][j]
    return C

# wyliczenie normy residuum z wektora
def norma_residuum(r):
    n = 0
    for i in range(len(r)):
        n += (r[i][0])**2

    wynik = math.sqrt(n)
    return wynik

# generacja macierzy trojkatnych dolnej L i górnej U z macierzy współczynników
def macierze_trojkatne_LU(A):
    L = zera_macierz(len(A), len(A[0]))
    U = zera_macierz(len(A), len(A[0]))

    # jedynki na przekatnej
    for w in range(len(A)):
        for k in range(len(A[0])):
            if (w == k):
                L[w][k] = 1

    # Metoda Doolittle’a
    for i in range(len(A)):

        for j in range(i, len(A)):
            sum1 = 0
            for k in range (0, i):
                sum1 += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - sum1

        for j in range(i+1, len(A)):
            sum2 = 0
            for k in range (0, i):
                sum2 += L[j][k] * U[k][i]
            L[j][i] = (1/U[i][i]) * (A[j][i] - sum2)

    return L, U

# generacja kopii macierzy
def przypisz_macierz_pomocnicza(A):
    B = zera_macierz(len(A), len(A[0]))
    for w in range(len(A)):
        for k in range(len(A[0])):
            B[w][k] = A[w][k]
    return B

