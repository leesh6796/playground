#include <iostream>
#include <memory>
using std::cout;
using std::endl;

class Animal
{
    public:
        virtual void speak() = 0; // pure virtual function. pure virtual function을 가진 클래스는 object로 만들 수 없다. 또한, pure virtual function을 override 하지 않으면 에러가 발생한다.
        virtual ~Animal()=default; // Base class의 destructor는 virtual public으로 만들어주어야 한다. 안그러면 child class의 destructor가 호출되지 않는다.
};

class Cat : public Animal
{
    public:
        void speak() override
        {
            cout << "meow~" << endl;
        }
};

class Dog : public Animal
{
    public:
        void speak() override
        {
            cout << "bark!" << endl;
        }
};


int main()
{
    std::array<std::unique_ptr<Animal>, 5> animals;
    
    for(auto & animalPtr : animals)
    {
        int i = 0;
        std::cin >> i;

        // dynamic polymorphism. runtime에 어떤 type으로 만들어질지 dynamic하게 결정된다.
        if(i == 1) animalPtr = std::make_unique<Cat>();
        else animalPtr = std::make_unique<Dog>();
    }

    for(auto & animalPtr : animals)
    {
        animalPtr->speak();
    }

    return 0;
}
