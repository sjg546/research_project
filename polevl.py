def polevl(x, coefs, N):
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        ans += coef * x**power
        power -= 1
    return ans
    
def p1evl(x, coefs, N):
    return polevl(x, [1] + coefs, N)
