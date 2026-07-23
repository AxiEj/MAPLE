##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Thu Jul 23 01:58:18 2026
##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path born.pqr
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing nlev...
PBEparm_parseToken:  trying nlev...
MGparm_parseToken:  trying nlev...
NOsh_parseMG:  Parsing grid...
PBEparm_parseToken:  trying grid...
MGparm_parseToken:  trying grid...
NOsh_parseMG:  Parsing gcent...
PBEparm_parseToken:  trying gcent...
MGparm_parseToken:  trying gcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing lpbe...
PBEparm_parseToken:  trying lpbe...
NOsh: parsed lpbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 0.
NOsh:  nlev = 4, dime = (97, 97, 97)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing nlev...
PBEparm_parseToken:  trying nlev...
MGparm_parseToken:  trying nlev...
NOsh_parseMG:  Parsing grid...
PBEparm_parseToken:  trying grid...
MGparm_parseToken:  trying grid...
NOsh_parseMG:  Parsing gcent...
PBEparm_parseToken:  trying gcent...
MGparm_parseToken:  trying gcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing lpbe...
PBEparm_parseToken:  trying lpbe...
NOsh: parsed lpbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 0.
NOsh:  nlev = 4, dime = (97, 97, 97)
NOsh: Done parsing ELEC section (nelec = 2)
NOsh: Parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 1 atoms
Valist_getStatistics:  Max atom coordinate:  (0, 0, 0)
Valist_getStatistics:  Min atom coordinate:  (0, 0, 0)
Valist_getStatistics:  Molecule center:  (0, 0, 0)
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 0 (1)
NOsh_setupCalc:  Mapping ELEC statement 1 (2) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 3
Vpbe_ctor2:  solute dimensions = 6 x 6 x 6
Vpbe_ctor2:  solute charge = 1
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 12 x 12 x 12 table
Vclist_ctor2:  Using 12 x 12 x 12 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (13.916, 13.916, 13.916)
Vclist_setupGrid:  Grid lower corner = (-6.958, -6.958, -6.958)
Vclist_assignAtoms:  Have 1000 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 301.719
Vacc_storeParms:  Using 3064-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.000150
Vpmg_fillco:  done filling coefficient arrays
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 7.589660e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (097, 097, 097)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 2.899300e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (049, 049, 049)
Vbuildops: Galer: (025, 025, 025)
Vbuildops: Galer: (013, 013, 013)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 9.902760e-01
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.156694e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.240935e-01
Vprtstp: contraction number = 1.240935e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.634257e-02
Vprtstp: contraction number = 1.316956e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 3.240369e-03
Vprtstp: contraction number = 1.982778e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 9.124897e-04
Vprtstp: contraction number = 2.816005e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 3.020560e-04
Vprtstp: contraction number = 3.310240e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.053364e-04
Vprtstp: contraction number = 3.487312e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 3.726395e-05
Vprtstp: contraction number = 3.537615e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 1.323368e-05
Vprtstp: contraction number = 3.551336e-01
Vprtstp: iteration = 9
Vprtstp: relative residual = 4.704392e-06
Vprtstp: contraction number = 3.554863e-01
Vprtstp: iteration = 10
Vprtstp: relative residual = 1.672769e-06
Vprtstp: contraction number = 3.555759e-01
Vprtstp: iteration = 11
Vprtstp: relative residual = 5.948295e-07
Vprtstp: contraction number = 3.555958e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 1.839904e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.979398e+01
Vpmg_setPart:  lower corner = (-15.84, -15.84, -15.84)
Vpmg_setPart:  upper corner = (15.84, 15.84, 15.84)
Vpmg_setPart:  actual minima = (-15.84, -15.84, -15.84)
Vpmg_setPart:  actual maxima = (15.84, 15.84, 15.84)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 2.089344161347E+03 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 1.684000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 3
Vpbe_ctor2:  solute dimensions = 6 x 6 x 6
Vpbe_ctor2:  solute charge = 1
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 12 x 12 x 12 table
Vclist_ctor2:  Using 12 x 12 x 12 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (13.916, 13.916, 13.916)
Vclist_setupGrid:  Grid lower corner = (-6.958, -6.958, -6.958)
Vclist_assignAtoms:  Have 1000 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 301.719
Vacc_storeParms:  Using 3064-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 3.806910e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (097, 097, 097)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 2.128900e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (049, 049, 049)
Vbuildops: Galer: (025, 025, 025)
Vbuildops: Galer: (013, 013, 013)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 6.709110e-01
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.211495e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.132480e-01
Vprtstp: contraction number = 1.132480e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.184172e-02
Vprtstp: contraction number = 1.045645e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 1.262676e-03
Vprtstp: contraction number = 1.066294e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 1.354407e-04
Vprtstp: contraction number = 1.072649e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 1.460124e-05
Vprtstp: contraction number = 1.078054e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.584350e-06
Vprtstp: contraction number = 1.085079e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 1.728125e-07
Vprtstp: contraction number = 1.090747e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 1.007990e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.118490e+01
Vpmg_setPart:  lower corner = (-15.84, -15.84, -15.84)
Vpmg_setPart:  upper corner = (15.84, 15.84, 15.84)
Vpmg_setPart:  actual minima = (-15.84, -15.84, -15.84)
Vpmg_setPart:  actual maxima = (15.84, 15.84, 15.84)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 2.274572488122E+03 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 2.084000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
printEnergy:  Performing global reduction (sum)
Vcom_reduce:  Not compiled with MPI, doing simple copy.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 3.240460e+01
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Thu Jul 23 01:58:20 2026
##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
