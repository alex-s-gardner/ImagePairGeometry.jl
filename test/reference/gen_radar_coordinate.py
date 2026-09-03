"""Generate the RadarCoordinate fixture: footprint bounding box and scene-center incidence angle.

Runs GeogridRadar.determineBbox and GeogridRadar.getIncidenceAngle as the reference defines them —
the same isce3.geometry.rdr2geo calls, in the same order, over the same sample grid — against a
synthetic circular orbit the Julia side rebuilds from the recorded analytic parameters.

The "projection" is lon/lat in degrees rather than a real CRS, so the comparison isolates the radar
solve from PROJ. A real reprojection is exercised by the whole-kernel fixture instead.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_radar_coordinate.py
"""
import json, numpy as np, isce3, copy
from isce3.core import Ellipsoid, Orbit, StateVector, DateTime, LookSide
from isce3.geometry import rdr2geo, DEMInterpolator

A=6378137.0; E2=0.0066943799901
EPOCH = DateTime('2021-06-15T00:00:00.000000')
el = Ellipsoid(a=A, e2=E2)
h = lambda x: {'dec': float(x), 'hex': float(x).hex()}

R=7.0e6; mu=3.986004418e14; w=np.sqrt(mu/R**3); inc=np.deg2rad(98.0)
n=61; spacing=10.0
times=[(i)*spacing for i in range(n)]
svs=[]
for ti in times:
    a=w*ti
    p=np.array([R*np.cos(a), R*np.sin(a)*np.cos(inc), R*np.sin(a)*np.sin(inc)])
    v=np.array([-R*w*np.sin(a), R*w*np.cos(a)*np.cos(inc), R*w*np.cos(a)*np.sin(inc)])
    svs.append(StateVector(EPOCH + isce3.core.TimeDelta(ti), p, v))
orb = Orbit(svs, EPOCH)

prf=486.4863103; dr=2.329562114715323; sr=8.0e5; wvl=0.05546576
nsamp=10000; nlin=8000; aztime=300.0
side=LookSide.Right
zrange=[-200.0,4000.0]
deg2rad=np.pi/180.0

# determineBbox, transcribed from GeogridRadar.py:140-251 with an identity lon/lat "projection"
rng = sr + np.linspace(0, nsamp-1, num=21)*dr
deltat = np.linspace(0,1.0,num=21)[1:-1]
llhs=[]
for rr in rng:
    for zz in zrange:
        llhs.append(rdr2geo(aztime, rr, orb, side, 0.0, wvl, DEMInterpolator(zz), el))
sensingStop = aztime + (nlin-1)/prf
for rr in rng:
    for zz in zrange:
        llhs.append(rdr2geo(sensingStop, rr, orb, side, 0.0, wvl, DEMInterpolator(zz), el))
for frac in deltat:
    st = aztime + frac*(nlin-1)/prf
    for rr in [rng[0], rng[-1]]:
        for zz in zrange:
            llhs.append(rdr2geo(st, rr, orb, side, 0.0, wvl, DEMInterpolator(zz), el))
L=np.array([np.array(x).ravel() for x in llhs])
lon=L[:,0]/deg2rad; lat=L[:,1]/deg2rad

# getIncidenceAngle, GeogridRadar.py:253-297
midrng = sr + (np.floor(nsamp/2)-1)*dr
midsensing = aztime + (np.floor(nlin/2)-1)/prf
mpos,_ = orb.interpolate(midsensing)
mxyz=np.array(mpos).ravel()
thetas=[]
for zz in zrange:
    llh=np.array(rdr2geo(midsensing, midrng, orb, side, 0.0, wvl, DEMInterpolator(zz), el)).ravel()
    targ=np.array(el.lon_lat_to_xyz(llh).tolist()).ravel()
    los=(mxyz-targ)/np.linalg.norm(mxyz-targ)
    nv=np.array([np.cos(llh[1])*np.cos(llh[0]), np.cos(llh[1])*np.sin(llh[0]), np.sin(llh[1])])
    thetas.append(np.arccos(np.dot(los,nv)))
ia=float(np.mean(thetas))

out={'orbit':{'n':n,'spacing':h(spacing),'t0':h(0.0),'R':h(R),'w':h(w),'inc':h(inc)},
     'radar':{'prf':h(prf),'dr':h(dr),'starting_range':h(sr),'wavelength':h(wvl),
              'nsamples':nsamp,'nlines':nlin,'sensing_start':h(aztime),'look_side':'right'},
     'bbox':{'lon':[h(lon.min()),h(lon.max())],'lat':[h(lat.min()),h(lat.max())],
             'npoints':len(llhs)},
     'incidence_angle':h(ia),
     'midrange':h(midrng),'midsensing':h(midsensing),
     'provenance':{'isce3_version':isce3.__version__}}
open('test/reference/radar_coordinate.json','w').write(json.dumps(out,indent=1,sort_keys=True))
print(f"bbox lon=({lon.min():.10f},{lon.max():.10f}) lat=({lat.min():.10f},{lat.max():.10f})")
print(f"npoints={len(llhs)}  incidence={np.rad2deg(ia):.10f} deg")
