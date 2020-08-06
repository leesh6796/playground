#include <iostream>
#include <memory>
using namespace std;

class Cat
{
    public:
        Cat() : mAge{0}
        {
            cout << "cat constructor" << endl;
        }
        ~Cat()
        {
            cout << "cat destructor" << endl;
        }

    private:
        int mAge;
};


int main()
{
    shared_ptr<Cat> catPtr = make_shared<Cat>();
    cout << "count: " << catPtr.use_count() << endl;

    shared_ptr<Cat> catPtrl = catPtr;
    cout << "count: " << catPtr.use_count() << endl;

    return 0;
}
