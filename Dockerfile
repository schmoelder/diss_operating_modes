

#
 
s
y
n
t
a
x
=
d
o
c
k
e
r
/
d
o
c
k
e
r
f
i
l
e
:
1




#
 
C
o
m
m
e
n
t
s
 
a
r
e
 
p
r
o
v
i
d
e
d
 
t
h
r
o
u
g
h
o
u
t
 
t
h
i
s
 
f
i
l
e
 
t
o
 
h
e
l
p
 
y
o
u
 
g
e
t
 
s
t
a
r
t
e
d
.


#
 
I
f
 
y
o
u
 
n
e
e
d
 
m
o
r
e
 
h
e
l
p
,
 
v
i
s
i
t
 
t
h
e
 
D
o
c
k
e
r
f
i
l
e
 
r
e
f
e
r
e
n
c
e
 
g
u
i
d
e
 
a
t


#
 
h
t
t
p
s
:
/
/
d
o
c
s
.
d
o
c
k
e
r
.
c
o
m
/
g
o
/
d
o
c
k
e
r
f
i
l
e
-
r
e
f
e
r
e
n
c
e
/




#
 
W
a
n
t
 
t
o
 
h
e
l
p
 
u
s
 
m
a
k
e
 
t
h
i
s
 
t
e
m
p
l
a
t
e
 
b
e
t
t
e
r
?
 
S
h
a
r
e
 
y
o
u
r
 
f
e
e
d
b
a
c
k
 
h
e
r
e
:
 
h
t
t
p
s
:
/
/
f
o
r
m
s
.
g
l
e
/
y
b
q
9
K
r
t
8
j
t
B
L
3
i
C
k
7




A
R
G
 
C
O
N
D
A
_
V
E
R
S
I
O
N
=
2
4
.
1
1
.
3


F
R
O
M
 
c
o
n
d
a
f
o
r
g
e
/
m
i
n
i
f
o
r
g
e
3
:
$
{
C
O
N
D
A
_
V
E
R
S
I
O
N
}
-
0
 
A
S
 
b
a
s
e




#
 
P
r
e
v
e
n
t
s
 
P
y
t
h
o
n
 
f
r
o
m
 
w
r
i
t
i
n
g
 
p
y
c
 
f
i
l
e
s
.


E
N
V
 
P
Y
T
H
O
N
D
O
N
T
W
R
I
T
E
B
Y
T
E
C
O
D
E
=
1




#
 
K
e
e
p
s
 
P
y
t
h
o
n
 
f
r
o
m
 
b
u
f
f
e
r
i
n
g
 
s
t
d
o
u
t
 
a
n
d
 
s
t
d
e
r
r
 
t
o
 
a
v
o
i
d
 
s
i
t
u
a
t
i
o
n
s
 
w
h
e
r
e


#
 
t
h
e
 
a
p
p
l
i
c
a
t
i
o
n
 
c
r
a
s
h
e
s
 
w
i
t
h
o
u
t
 
e
m
i
t
t
i
n
g
 
a
n
y
 
l
o
g
s
 
d
u
e
 
t
o
 
b
u
f
f
e
r
i
n
g
.


E
N
V
 
P
Y
T
H
O
N
U
N
B
U
F
F
E
R
E
D
=
1




W
O
R
K
D
I
R
 
/
r
d
m
_
w
o
r
k
d
i
r




U
S
E
R
 
r
o
o
t




#
 
P
r
e
v
e
n
t
s
 
i
n
t
e
r
a
c
t
i
v
e
 
p
r
o
m
p
t
s
 
d
u
r
i
n
g
 
a
p
t
-
g
e
t


A
R
G
 
D
E
B
I
A
N
_
F
R
O
N
T
E
N
D
=
n
o
n
i
n
t
e
r
a
c
t
i
v
e




R
U
N
 
a
p
t
-
g
e
t
 
u
p
d
a
t
e
 
&
&
 
a
p
t
-
g
e
t
 
i
n
s
t
a
l
l
 
-
y
 
g
i
t
 
g
i
t
-
l
f
s
 
s
s
h
 
&
&
 
 
 
 
 
a
p
t
-
g
e
t
 
c
l
e
a
n
 
&
&
 
r
m
 
-
r
f
 
/
v
a
r
/
l
i
b
/
a
p
t
/
l
i
s
t
s
/
*




C
O
P
Y
 
e
n
v
i
r
o
n
m
e
n
t
.
y
m
l
 
/
t
m
p
/
e
n
v
i
r
o
n
m
e
n
t
.
y
m
l




R
U
N
 
c
o
n
d
a
 
e
n
v
 
u
p
d
a
t
e
 
-
n
 
b
a
s
e
 
-
-
f
i
l
e
 
/
t
m
p
/
e
n
v
i
r
o
n
m
e
n
t
.
y
m
l


