#include <iostream>
using std::cout;
using std::endl;

struct complex
{
    double re;
    double im;

    complex(double r, double i) : re{r}, im{i} {};
    void print() const
    {
        cout << re << " " << im << "i" << endl;
    }
};

complex operator+(const complex& lhs, const complex& rhs)
{
    complex c{lhs.re + rhs.re, lhs.im + rhs.im};
    return c;
}

std::ostream& operator<<(std::ostream& os, const complex& c)
{
    return os << c.re << " " << c.im << "i";
}

int main()
{
    complex c1{1,1};
    complex c2{1,2};

    complex c{c1+c2};
    c.print();
    cout << c << endl;

    return 0;
}
