
# The Cognitive Rule Engine (CRE)


## A high performance rule engine for python. 
Is a forward-chaining (i.e. RETE-like) rule engine for Python. CRE utilizes [numba](https://numba.pydata.org/) a JIT compiler for Python to translate rules into optimized machine code at runtime. CRE rules can be written in Python and run with the performance of C code. 

**NOTE: CRE is a work in progress and currently only implements only a subset of its planned capabilities.**
- [x] C-like Data-Structures for Facts + Working Memory
- [x] Complex Pythonic Conditions
- [x] High-Performance Incremental Condition Matching 
- [ ] Pythonic Rule Specification
- [ ] High-Performance Rule Engine
- [x] Incremental Data Processing Utilities
- [x] Set Chaining Planner

## Installation



## Fact Types
Facts are struct-like objects that are declared to a `MemSet` (i.e. working-memory). Unlike normal Python objects facts are strongly typed and consists of a fixed number of typed attribute-value pairs. 

### `define_fact`
The layout for a new fact type can be defined with `define_fact`.

```python
from cre import define_fact
Element = define_fact("Element", {"id" : str, 'val' : int, 'next' : 'Element'})
```
Using `'inherit_from'` will make a fact type inherit from another, and facts types can even have fact-like (e.g. `next` above) or list-like (e.g. `children` below) members.   
```python
Container = define_fact("Container", {'inherit_from' : Element, 'children' : 'List(Element)'})
```
### Fact Instances
Internally a fact's members are allocated in a contiguous region of memory outside of Python. However, we can still instantiate and modify facts within native Python:

```python
a = Element('a', 1)             # positional arguments
b = Element(A=2,B='b',next=a)   # or key-word arguments
c = Element()                   # omitted args default => Element(id='', val=0, next=None)
p = Container('parent', 7, children=[a,b,c])
c.next = b                      # Can also assign afer instantiation
```

## MemSet (i.e. Working Memory) 

A `MemSet` is a container that holds facts. This is CRE's data-structure for working memory, but it also has a wider range of uses. 
### `declare` and `retract`
Facts can be added and removed from a MemSet using the `declare` and `retract` methods. 
```python
from cre import MemSet
wm = MemSet()
wm.declare(a)
wm.declare(b)
wm.declare(c)
wm.declare(p)
wm.retract(a)
print(wm)
```
Output:
```
MemSet(facts=(
    Element(id='b', val=2, next=<Element at 0x32ace50>)
    Element(id='', val=0, next=<Element at 0x32acf00>)
    Container(id='parent', val=7, next=None, children=List([<BaseFact at 0x32ace50>, <BaseFact at 0x32acf00>, <BaseFact at 0x32cded0>]))
)
----
```
### `idrec`
MemSets keep lightweight records of all changes made to them. Each chage has an associated  identification record or `idrec`, which is just a packed 64-bit uint consisting of three unsigned-values:
1. `t_id` (16-bit) identifies a type (usually a fact type).
2. `f_id` (40-bit) identifies the particular fact with `t_id` within a MemSet.
3. `a_id` (8-bit)  identifies an argument within a fact.

```python
from cre.utils import encode_idrec, decode_idrec
idrec = encode_idrec(20,1,0) # 5629499534213376
t_id, f_id, a_id = decode_idrec(idrec) # (20, 1, 0)
```
When a fact is declared it is assigned an `idrec` with a unique `f_id` and  `a_id=0`. `declare` will return the declared fact's idrec. Instead of keeping references to each fact we can use a fact's `idrec` with `get_fact` or `retract` to retreive or remove it from a memset.
```python
idrec = wm.declare(Element(id='z', val=9))
z = wm.get_fact(idrec)
wm.retract(idrec)
```
### `modify`
When a fact is declared its mutability becomes protected. If we try to set an attribute of a declared fact directly we'll get an `AttributeError` error. 
```python
wm.declare(c)
c.id = 'c'
```
Output:
```
AttributeError: Facts objects are immutable once declared. Use mem.modify instead.
```
Instead we can use the `modify` method of MemSet to change a fact. This helps MemSet to track all changes to the fact. 
```python
wm.modify(c, 'id', 'c')
print(c)
```
Output:
```
Element(id='c', val=0, next=<Element at 0x24b1be0>)
```
### `get_facts`
`get_facts` can be used to get all instances of a particular type of fact.
```python
for fact in wm.get_facts(Element):
  print(fact)
```
Output:
```
Element(id='c', val=0, next=<Element at 0x3e8cad0>)
Element(id='b', val=2, next=<Element at 0x3e8ca20>)
Element(id='c', val=0, next=<Element at 0x3e8cad0>)
Container(id='parent', val=7, next=None, children=List([<BaseFact at 0x3e8ca20>, <BaseFact at 0x3e8cad0>, <BaseFact at 0x3e6eff0>]))
```
By default get_fact will retrieve subtypes of the given fact type. For instance the 'parent' `Container` fact is included in the output above. To only get the `Element` facts we could add `nosubtypes=True` like so `wm.get_facts(Element, no_subtypes=True)`. To get all types regardless of type we can omit the type like `wm.get_facts()`.

## Conditions
We can define a set of `Conditions` that pick out sets objects in a `MemSet`.

### Var
`Var` is used in various places in CRE to define typed variables in declarative statements. For instance, when writing conditions each var `Var` indicates one fact we would like to extract during matching. Vars are provided a type and an optional `alias`.
```python
from cre import Var
A = Var(Element) # An unaliased Var
B = Var(Element, "B") # A Var aliased to "B"
P = Var(Container, "P") # A Var aliased to "P"
```
> :warning: Every `Var` instance is an independant object even if multiple instances are assigned the same alias. To avoid confusing print statements it is good practice to give every instance of `Var` a unique alias in the context of a single declarative statement (like within the same `Conditions` object).

### Writing Conditions
Conditions are logical statements in [Disjunctive Normal Form](https://en.wikipedia.org/wiki/Disjunctive_normal_form). In other on `Conditions` object is a disjunction of conjunctions of `Literal` statements. Most conditions can be built by simply writing python expressions `Literal`s encloded in `()` connected by `&` and `|' qualifiers for `AND` and `OR` respectively.

```python
c1 = (((A.id == "c") & (A.val >= 1)) 
c2 = (A.val < B.val) & (A.next.id == B.id) & (A.next == B) 
```
Each literal statement can contain one Var (an Alpha Literal) or two Vars (a Beta Literal), and a set of Conditions can contain an arbitrary number of Vars. 

### `get_matches`
We can then use `get_matches` to get all sets of facts that match these conditions. For instance lets build a new working memory and get the matches for `c1` and `c2` on it:
```python
wm = MemSet()
c = Element('c', 3)
b = Element('b', 2, next=c)
a = Element('a', 1, next=b)
wm.declare(a)
wm.declare(b)
wm.declare(c)
wm.declare(Container('parent', 7, children=[a,b,c]))

print("c1 matches:")
for match in c1.get_matches(wm):
    print(match)
    
print("c2 matches:")
for match in c2.get_matches(wm):
    print(match)
```
Output:
```
c1 matches:
(Element(id='c', val=3.0, next=None), Container(id='parent', val=7.0, next=None, children=List([<BaseFact at 0x2e1a040>, <BaseFact at 0x2e19f90>, <BaseFact at 0x2da9fe0>])))
c2 matches:
(Element(id='a', val=1.0, next=<Element at 0x2e19f90>), Element(id='b', val=2.0, next=<Element at 0x2da9fe0>))
(Element(id='b', val=2.0, next=<Element at 0x2da9fe0>), Element(id='c', val=3.0, next=None))
```

### `antiunify`
The `antiunify` method of a `Conditions` object can be used to to get the antiunion of two logical statements---essentially the intersection of two `Conditions` objects.  
```python
from numba import f8
x, y, z = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')
a, b, c, d = Var(f8,'a'), Var(f8,'b'), Var(f8,'c'), Var(f8,'d')
c1 = (x < y) & (y < z) & (y < z) & (z != x) & (y != 0) 
c2 = (a < b) & (b < c) & (b < c) & (b < c) & (c != a) & (b != 0) & (d != 0)
print(c1.antiunify(c2))
```
Output:
```
((x < y) & (y < z) & (y < z) & (z != x) & (y != 0))
```
In the example above the antiunion is expressed using the variables for `c1`. `antiunify` has found the optimal structure mapping between of variables of the two statements `x->a`, `y->b`, `z->c` and eliminated the variable `d'. Disjunctive statements can even be rearranged by this structure mapping.
```python
c1 = ((x < y) & (z != x) & (y != 0) | # 1
      (x < y) & (z == x) & (y != 7) | # 2
      (x > y) & (z != x) & (y != 2)   # 3
     )
c2 = ((a < b) & (c == a) & (b != 7) & (d > 0) | #2
      (a < b) & (c != a) & (b != 0) |           #1
      (a > b) & (c != a) & (b != 0) & (d != 7)  #3
     )
print(c1.antiunify(c2))
```
Output:
```
((x < y) & ~(z == x) & ~(y == 0) |\
 (x < y) & (z == x) & ~(y == 7) |\
 (x > y) & ~(z == x))
```

## Op
`Op` is a special declarative function used by CRE for a number of purposes. For instance, each `Literal` in a `Conditions` object has an underlying `Op` that is executed to check candidate facts or pair of facts. Custom `Op` types can be defined by the user.
```python
