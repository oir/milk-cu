## milk

Milk is a small GPU library for neural nets with a focus on more NLP oriented stuff such as recurrent and recursive nets. Based on [mshadow](https://github.com/dmlc/mshadow). It's main goal is to unify my own research code so it is easier to transfer knowledge and models between different research projects.

You can find a guide [here](https://github.com/oir/milk-cu/blob/master/guide/guide.md).

Feel free to ask questions: oirsoy [a] cs [o] cornell [o] edu.

## Some examples

See all the details in the appropriate files under the examples folder.

A simple three layer feedforward net (`examples/mnist.cu`):

    auto ds = datastream(2);
    auto nn = ff(100,nonlin::tanh()) >>
              ff(100,nonlin::tanh()) >>
              ff(100,nonlin::tanh()) >>
              ff(10,nonlin::id());
    auto loss = smax_xent();
    auto all = ds >> nn >> loss;

An LSTM as a 5-class classifier for sentences (`examples/sstb-lstm.cu`):

    auto ds = datastream(2);
    auto wv = proj(300, N+1);
    auto nn = ds >> wv >> lstm(50) >> tail() >> ff(5,nonlin::id()) >> smax_xent();

A deep bidirectional recurrent net for opinion mining (Irsoy & Cardie, EMNLP2014, `examples/mpqa.cu`):

    auto ds = datastream(2);
    auto wv = proj(300, nX+1);
    auto layer = []() { return cast() >>
                               (recurrent(100), recurrent(100, reverse)) >>
                               cat(); };
    auto top = smax_xent();
    auto nn = ds >> wv >> layer() >> layer() >> layer()
                 >> ff(3,nonlin::id()) >> top;

A recursive net (e.g. Socher et al, ICML2011 or Iyyer et al, EMNLP2014):

    auto ds = datastream(2);
    auto wv = proj(300, N+1);
    auto nn = ds >> wv >> recursive(100, 2) >> ff(5,nonlin::id()) >> smax_xent();

Or a deep (three-layer) rectifier recursive net (Irsoy & Cardie, NIPS2014, `examples/sstb-rsv.cu`):

    auto ds = datastream(2);
    auto wv = proj(300, N+1);
    auto top = smax_xent();
    auto nn = ds >> wv >> recursive(50, 2, nonlin::relu()) >>
                          recursive(50, 2, nonlin::relu()) >>
                          recursive(50, 2, nonlin::relu()) >>
    ff(5,nonlin::id()) >> top;

## Todo (at a high level)

* Add a tree LSTM model
* Add budding trees
* Add a recursive net example with dependency trees instead of constituency trees.

## License

Code is released under [the Apache v2 license](https://github.com/oir/milk-cu/blob/master/LICENSE).
