def spinSz_from_sz_mass(s1z,s2z,mass1,mass2):
    return (mass1*s1z + mass2*s2z)/(mass1 + mass2)

def spinAz_from_sz_mass(s1z,s2z,mass1,mass2):
    return (mass1*s1z - mass2*s2z)/(mass1 + mass2)

def s1z_from_spinSz_spinAz(spinSz,spinAz,mass1,mass2) :
    return (spinSz + spinAz) * ((mass1 + mass2)/(2*mass1))

def s2z_from_spinSz_spinAz(spinSz,spinAz,mass1,mass2) :
    return (spinSz - spinAz) * ((mass1 + mass2)/(2*mass2))