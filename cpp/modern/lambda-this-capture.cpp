#include <iostream>
using namespace std;

class Cat
{
    public:
        explicit Cat(int age) : mAge(age) { }
        void speak() const
        {
            cout << "meow" << endl;
        }
        void test() const
        {
            int num = 5;
            auto lambda = [this, num]() // lambda this capture to use member variables in lambda function
            {
                cout << "lambda function" << endl;
                cout << this->mAge << endl;
                cout << num << endl;
                this->speak();
            };
            lambda();
        }

    private:
        int mAge;
};


int main()
{
    Cat kitty{1};
    kitty.test();

    return 0;
}
