import numpy as np
import pd.particles as particles
import itertools

a0 = 0.52917721092
class Cell( np.ndarray ):

    def __new__(cls, 
            my_min = [0.0, 0.0, 0.0],
            my_max = [10.0, 10.0, 10.0],
            my_cutoff = 1.5,
            AA = False,
        ):

        xdim = int( np.ceil ( (my_max[0] - my_min[0])/my_cutoff ))
        ydim = int( np.ceil ( (my_max[1] - my_min[1])/my_cutoff ))
        zdim = int( np.ceil ( (my_max[2] - my_min[2])/my_cutoff ))

        if xdim == 0:
            xdim = 1
        if ydim == 0:
            ydim = 1
        if zdim == 0:
            zdim = 1
        shape = (xdim, ydim, zdim)
        obj = np.zeros(shape, dtype = object ).view(cls)
        return obj

    def __init__(self, 
            my_min = [0.0, 0.0, 0.0],
            my_max = [10.0, 10.0, 10.0],
            my_cutoff = 1.5,
            AA = False):
        """docstring for __init__"""

        self.AA = AA

        self.my_xmin = my_min[0]
        self.my_ymin = my_min[1]
        self.my_zmin = my_min[2]
        
        self.my_xmax = my_max[0]
        self.my_ymax = my_max[1]
        self.my_zmax = my_max[2]

        self.my_cutoff = my_cutoff

        self.xdim = int( np.ceil ( (self.my_xmax - self.my_xmin)/my_cutoff ))
        self.ydim = int( np.ceil ( (self.my_ymax - self.my_ymin)/my_cutoff ))
        self.zdim = int( np.ceil ( (self.my_zmax - self.my_zmin)/my_cutoff ))

        if self.xdim == 0:
            self.xdim = 1
        if self.ydim == 0:
            self.ydim = 1
        if self.zdim == 0:
            self.zdim = 1

        tmp = self.ravel()

        tmp[:] = [[] for i in range(self.xdim) 
            for j in range(self.ydim) for k in range(self.zdim)]
        self[:] = tmp.reshape( ( self.xdim, self.ydim, self.zdim ))

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @staticmethod
    def from_PointDipoleList( pdl, co = 25.0 ):
        """By default, the cutoff box is 25 Angstroms"""

        co /= a0
        x, y, z = [], [], []
        for p in pdl:
            x.append( p._r[0] )
            y.append( p._r[1] )
            z.append( p._r[2] )

        cell = Cell( my_min = [ np.min(x), np.min(y), np.min(z )],
                    my_max = [ np.max(x), np.max(y), np.max(z )],
              my_cutoff = co,
              )
        for pd in pdl:
            cell.add(pd)
        return cell

    def add(self, item ):
        x_ind, y_ind, z_ind = self.get_index( item )
        if item not in self[ x_ind, y_ind, z_ind ]:
            self[ x_ind, y_ind, z_ind ].append( item )

    def __iter__(self):
        for i in range(len(self)):
            for j in range(len(self[i])):
                for k in range(len(self[i][j])):
                    for at in self[i][j][k]:
                        yield at
    def get_closest( self, item ):
            """
    Return the closest items, 
    to iterate not over whole cell box but closest

        >>> c = Cell( my_cutoff = 1.5 )
        >>> a1 = Atom( element = 'H', x = 1.4 ) #in the x index 0
        >>> a2 = Atom( element = 'O', x = 1.6 ) #in the x index 1
        >>> c.add( a1 )
        >>> c.add( a2 )
        >>> c.get_closest( a1 ) #Will return list where only the Oxygen exists
        [<molecules.Atom at 0x0xah5ah3h5] 
            """
            x_ind, y_ind, z_ind = self.get_index( item )
            tmp_list = []
            new =  []

            if x_ind == 0:
                if (self.shape[0] - 1 ) == x_ind:
                    xmin, xmax = 0, 1
                else:
                    xmin, xmax = 0, 2
            else:
                if (self.shape[0] - 1) == x_ind:
                    xmin, xmax = x_ind - 1, x_ind + 1
                else:
                    xmin, xmax = x_ind - 1, x_ind + 2

            if y_ind == 0:
                if (self.shape[1] - 1 ) == y_ind:
                    ymin, ymax = 0, 1
                else:
                    ymin, ymax = 0, 2
            else:
                if (self.shape[1] - 1) == y_ind:
                    ymin, ymax = y_ind - 1, y_ind + 1
                else:
                    ymin, ymax = y_ind - 1, y_ind + 2

            if z_ind == 0:
                if (self.shape[2] - 1 ) == z_ind:
                    zmin, zmax = 0, 1
                else:
                    zmin, zmax = 0, 2
            else:
                if (self.shape[2] - 1) == z_ind:
                    zmin, zmax = z_ind - 1, z_ind + 1
                else:
                    zmin, zmax = z_ind - 1, z_ind + 2

            for i, j, k in itertools.product( range( xmin, xmax ), range( ymin, ymax ), range( zmin, zmax )):
                        new += self[i, j, k] 
            new.remove( item )
            return new

    def get_index( self, item ):
        """
Return the x, y, and z index for cell for this item,

    >>> c = Cell( my_cutoff = 1.5 )
    >>> a1 = Atom( element = 'H', x = 1.4 ) #in the x index 0
    >>> print c.get_index( a1 )
    (0, 0, 0,)
"""
        if isinstance( item, particles.PointDipole ):
            x, y, z = item._r
            assert self.my_xmin <= x <= self.my_xmax
            assert self.my_ymin <= y <= self.my_ymax
            assert self.my_zmin <= z <= self.my_zmax

        tmp_xmin = x - self.my_xmin
        tmp_ymin = y - self.my_ymin
        tmp_zmin = z - self.my_zmin

        x_ind = int( np.floor( tmp_xmin /  self.my_cutoff))
        y_ind = int( np.floor( tmp_ymin /  self.my_cutoff))
        z_ind = int( np.floor( tmp_zmin /  self.my_cutoff))
        
        return (x_ind, y_ind, z_ind)
