import numpy as np
import json
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from math import sqrt, exp, pi, cos

am2au = 1822.88742
au2ar = 0.52917706
cm2au = 4.5554927e-6

def encode (i_list,cum_prod,nlen):
    i_rank = 0
    for i in xrange(nlen-1):
        i_rank += i_list[i]*cum_prod[i]
    i_rank += i_list[-1]
    return i_rank

def decode (i_rank,cum_prod,nlen):
    i_list = []
    for i in xrange(nlen-1):
        i_list.append(i_rank/cum_prod[nlen-i-2])
        i_rank %= cum_prod[nlen-i-2]
    i_list.append(i_rank)
    return i_list

def numloc_diff (i_list,j_list,nlen):
    num, loc = 0, -1
    for i in xrange(nlen):
        if i_list[i] != j_list[i]:
            loc  = i
            num += 1
            if num>1:
                return 2, -1
    return num, loc

def get_pot_1d (de, beta, r0, r):
    return de*(1-exp(-beta*(r-r0)))**2

def get_pot_2d (gxy, x, y):
    return gxy*x*y*(x+y)

def get_ham_1d (m):

    m0, ri, rf, de, be, re, nr = m['m0'], m['ri'], m['rf'], m['de'], m['be'], m['re'], m['nr']

    nk, dr, dk = (nr-1)/2, (rf-ri)/nr, 2.0*pi/(rf-ri)
    ttmp = dk*dk/(2.0*m0)
    ham = np.zeros((nr,nr),dtype=float)
    for i in xrange(nr):
        for j in xrange(nr):
            for k in xrange(nk):
                tk = 2.0*pi*(k+1)*(i-j)/nr
                ham[i,j] += cos(tk)*ttmp*(k+1)*(k+1)*2.0/nr
        ham[i,i] += get_pot_1d(de,be,re,ri+i*dr)
    return ham

def savelog(dic):
    for i in dic:
        with open(i,'w') as f:
            np.savetxt(f,dic[i])

def main ():

    with open('input.json','r') as f:
        d = json.load(f)
    ne = d['ne']
    modes = d['modes']
    gxy = d['gxy']

    for m in modes:
        m['m0'] *= am2au
        m['ri'] /= au2ar
        m['rf'] /= au2ar
        m['de'] *= cm2au
        m['be'] *= au2ar
        m['re'] /= au2ar
    vmn = [[v*cm2au for v in colv] for colv in gxy]

    n_list = np.array([m['nr'] for m in modes])
    ndim = np.prod(n_list)
    nmod = len(modes)

    cum_prod = np.zeros(nmod,dtype=int)
    cum_prod[0] = modes[nmod-1]['nr']
    for i in xrange(1,nmod):
        cum_prod[i] = cum_prod[i-1]*modes[nmod-i-1]['nr']

    hams = []
    for m in modes:
        hams.append(get_ham_1d(m))

    row, col, val = [], [], []
    for i in xrange(ndim):
        i_list = decode(i,cum_prod,nmod)
        for j in xrange(i,ndim):
            j_list = decode(j,cum_prod,nmod)
            if i != j:
                num, loc = numloc_diff(i_list,j_list,nmod)
                if num==1:
                    tmp = hams[loc][i_list[loc],j_list[loc]]
                    row.append(i)
                    col.append(j)
                    val.append(tmp)
                    row.append(j)
                    col.append(i)
                    val.append(tmp)
            else:
                tmp = 0.0
                for m in xrange(nmod):
                    rm = modes[m]['ri']+i_list[m]*(modes[m]['rf']-modes[m]['ri'])/modes[m]['nr']
                    tmp += hams[m][i_list[m],i_list[m]]
                    for n in xrange(m+1,nmod):
                        rn = modes[n]['ri']+i_list[n]*(modes[n]['rf']-modes[n]['ri'])/modes[n]['nr']
                        tmp += get_pot_2d(vmn[m][n],rm,rn)
                row.append(i)
                col.append(j)
                val.append(tmp)

    print 'NO of nonzero elements:', len(row)
    print 'sparse:', float(len(row))/float(ndim**2)

    hamt = coo_matrix((val,(row,col))).tocsc()

   # Arpack is not efficient for finding smallest eigvals,
   # so it's better to comment the line below and use shift-invert
   #energies, wavefunc = eigsh(hamt,ne,which='SM')
    energies, wavefunc = eigsh(hamt,ne,sigma=0,which='LM')
    savelog({'energies.dat':energies,'wavefunc.dat':wavefunc})
    print energies

   # Check orthogonal
    if any(abs(np.dot(wavefunc.T,wavefunc)-np.eye(ne))>1.e-12):
        raise ValueError("Wavefunction is not orthogonal!")

    for imod in xrange(nmod):
        m = modes[imod]
        q = np.empty(ndim,dtype=float)
        for i in xrange(ndim):
            i_list = decode(i,cum_prod,nmod)
            q[i] = m['ri']+i_list[imod]*(m['rf']-m['ri'])/m['nr']-m['re']
        qwave = np.array([[q[i]*wavefunc[i,j] for j in xrange(ne)] for i in xrange(ndim)])
        qmod = np.dot(wavefunc.T,qwave)
        savelog({'qmod_%d'%imod:qmod})


if __name__ == '__main__':
    main()