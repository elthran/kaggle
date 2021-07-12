# 1 - sin curve fitting

def objective(x, a, b, c, d):
    return a * np.sin(b - x) + c * x ** 2 + d


for cohort in roas_df['install_cohort'].unique():
    filtered_df = roas_df[roas_df['install_cohort'] == cohort].groupby("days_since_install").agg(
        roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()

    x = filtered_df["days_since_install"]
    y = filtered_df["roas"]

    popt, _ = curve_fit(objective, x, y)
    # summarize the parameter values
    a, b, c, d = popt
    print(popt)
    # plot input vs output
    plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), 1)
    # calculate the output for the range
    y_line = objective(x_line, a, b, c, d)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='red')
    plt.title(cohort)
    plt.show()











def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


gmodel = Model(gaussian)
result = gmodel.fit(y, x=X, amp=5, cen=5, wid=1)

print(result.fit_report())

plt.plot(X, y, 'bo')
plt.plot(X, result.init_fit, 'k--', label='initial fit')
plt.plot(X, result.best_fit, 'r-', label='best fit')
plt.legend(loc='best')
plt.show()















def func1(x, a, b, c):
    return a*x**2+b*x+c

def func2(x, a, b, c):
    return a*x**3+b*x+c

def func3(x, a, b, c):
    return a*x**3+b*x**2+c

def func4(x, a, b, c):
    return a*np.exp(b*x)+c

params, _ = curve_fit(func1, X, y)
a, b, c = params[0], params[1], params[2]
yfit1 = func1(X, a, b, c)


params, _ = curve_fit(func2, X, y)
a, b, c = params[0], params[1], params[2]
yfit2 = func1(X, a, b, c)


params, _ = curve_fit(func3, X, y)
a, b, c = params[0], params[1], params[2]
yfit3 = func1(X, a, b, c)


params, _ = curve_fit(func4, X, y)
a, b, c = params[0], params[1], params[2]
yfit4 = func1(X, a, b, c)

plt.plot(X, y, 'bo', label="y-original")
plt.plot(X, yfit1, label="y=a*x^2+b*x+c")
plt.plot(X, yfit2, label="y=a*x^3+b*x+c")
plt.plot(X, yfit3, label="y=a*x^3+b*x^2*c")
# plt.plot(X, yfit4, label="y=a*exp(b*x)+c")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.show()