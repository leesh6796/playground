#include <iostream>
#include <string>

using std::cout;
using std::endl;

class Cat
{
    public:
        Cat()=default; // default constructor는 컴파일러가 만들어주는 constructor를 기본으로 사용한다.
        Cat(std::string name, int age)
            : mName{std::move(name)}, mAge{age} // { }를 사용하는 것을 추천한다.
        {
            cout << mName << " constructor" << endl;
        }

        ~Cat() noexcept // Destructor
        {
            cout << mName << " destructor" << endl;
        }

        Cat(const Cat& other) : mName{other.mName}, mAge{other.mAge}
        {
            cout << mName << " copy constructor" << endl;
        }

        // destructor, move constructor는 noexcept를 붙여주어야 한다.
        Cat(Cat&& other) noexcept : mName{std::move(other.mName)}, mAge{other.mAge}
        {
            cout << mName << " move constructor" << endl;
        }

        Cat& operator=(const Cat& other)
        {
            mName = other.mName;
            mAge = other.mAge;
            cout << mName << " copy assignment" << endl;
            return *this;
        }

        Cat& operator=(Cat&& other)
        {
            mName = std::move(other.mName);
            mAge = other.mAge;
            cout << mName << " move assignment" << endl;
            return *this;
        }

        void print()
        {
            cout << mName << " " << mAge << endl;
        }

    private:
        std::string mName;
        int mAge;
};


int main()
{
    Cat kitty{"kitty", 1};
    
    Cat kitty2{kitty};
    Cat kitty3{std::move(kitty)};

    Cat nabi{"nabi", 2};
    kitty = nabi;
    kitty.print();

    kitty = std::move(nabi);

    return 0;
}
