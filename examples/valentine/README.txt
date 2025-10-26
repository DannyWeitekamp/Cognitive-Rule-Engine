---  To run OPS5: ---

Install a LISP implementation such as `sbcl`

Install quicklisp (https://www.quicklisp.org/) and ensure that it is set up with your LISP installation.

clone the OPS5 package (https://github.com/briangu/OPS5) into ~/quicklisp/local-projects

Then run:

	sbcl --load run_ops5_valentine.lisp


--- To run SOAR: ---

Download SOAR, alias the SoarCLI.sh to `soar-cli`
Edit valentines.soar to adjust the number of valentines and working memory size
Then run:

    `soar-cli -s valentines.soar`

With the SOAR cli still open run:

	`stat`

to see the total runtime


--- To run CORGI: ---

Install CRE by going to Cognitive-Rule-Engine/ and running `pip install -e .`

Then run:
	`python valentines.py -v 2 -r 3

Where -v set the number of valentines and -r is the number of repetitions of the 
dataset to produce working memory sizes of 50, 150, 200, etc...

