# Abstractions and Conventions

#### Layers

Main abstraction unit is a `layer` which contains:
* Intermediate and final data for computation
* Pointers to input data
* Weights belonging to input connections
* Necessary methods for forward & backward propagation

So for example the simple feedforward layer `ff` has:
```C++
Data<xpu> h;
Input<xpu> x;     // e.g. can point to `h' of some other ff layer, or input data
Weight<xpu> W, b;
void forward();   // e.g. h = nonlin(Wx + b)
void backward();
```

respectively.

This is the ultimate minimum necessary for computation.

#### Data, Input, Weight

These are the main objects to do math on.

A `Data` is essentially a set of two (pointers to*) matrices for the actual data _x_ and its gradient _dE/dx_:
```C++
Data<xpu> x;
x();    // gives you x
x.d();  // gives you its grad
```

(* I found this indirection easier since default copy constructor works out of the box and we can use `vector`s of `Data` easily, shuffle things, etc.)

An `Input` is almost a `Data*` with some additional stuff that makes things easier (for instance, you can still call `x()` and don't have to do `(*x)()` which is way uglier).
```C++
Input<xpu> x;
Data<xpu> y;
x.connect_from(y); // effectively x = &y
```

A `Weight` inherits from `Data`. It has additional weight related things such as an `updater` (which has update methods and also keeps the update history for e.g. adagrad) or `initer`.

#### Factory of layers

Every layer is defined in the namespace `layer`. Under the namespace `factory`, there are functions with the same names that construct and return a `shared_ptr` to the layer. For instance, `factory::ff(50)` returns a `shared_ptr<layer::ff>`. With this it is easy to name both subnetworks and entire networks and use them with container layers (below).

#### Container layers

So far I have two container layers for ease of use when combining different layers. `stack` to combine vertically, and `join` to combine horizontally (side-by-side). You can use shared pointers with factory semantics (see above) and `operator>>` and `operator,` to do things like:
```C++
auto nn1 = ff(50) >> ff(50) >> ff(50); // 3-layer feedforward
auto nn2 = cast() >> (recurrent(50), recurrent(50, reverse) >> cat();
```
When you use `stack` to combine two layers, connections are made using `ins()` and `outs()` methods of a layer object, one-by-one. For instance
```C++
auto bottom = ff(50);
auto top = ff(50);
auto s = bottom >> top;
```
constructs the stack `s` as well as calls `top->x.connect_from(bottom->h)`, since `bottom->outs()` returns the single element vector `{bottom->h}`, and `top->ins()` returns `{top->x}`.

`cast` and `cat` above is used to broadcast one input into multiple outputs, and to concatenate multiple inputs into a single output, respectively.

Being able to name both a subnetwork and the whole network makes it easier to make modifications to parts:
```C++
auto wv = proj(300, nwords); // word vector table
auto all = wv >> lstm(100) >> lstm(50) >> lstm(25);
all->set_updater<rmsprop<gpu>>();
all->set_lr(1e-3); // set learning rate of the entire thing
wv->set_lr(0.);    // override learning rate of word vector table
```

#### Whitebox

**You do not have to use any of the factory functions or container layers mentioned above if you don't want to, or what you want to do is nontrivial.** 

For instance, instead of 
```C++
using namespace milk::factory;
auto nn = ff(30) >> ff(40) >> ff(50); // nn is a shared_ptr
nn->forward();
```
you can do
```C++
using namespace milk::layer;
ff<gpu> layer1(30), layer2(40), layer3(50); // this are actual ff objects
layer2.x.connect_from(layer1.h);
layer3.x.connect_from(layer2.h);
for (ff<gpu>* l : {&layer1, &layer2, &layer3})
    l->forward();
```

and it should achieve the same effect with no use of the factory namespace functions or any container layer. Every attribute is public to hack into and do things manually.

# Sequences and Structures (aka recurrent and recursive)

TODO.
