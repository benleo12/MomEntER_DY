// Shower a POWHEG-BOX Z LHE with the STANDARD POWHEG+Pythia8 matching:
// pTmaxMatch=2 (full phase space) + the real PowhegHooks veto (main164 recipe,
// HED=201: pThard=2, pTemt=0, pTdef=1).  Clean Monash tune, no graft.
// Output per event (m_ll>40): qT, m_ll, dphi(raw), phi*_eta, lepton pT/eta, weight.
//
//   pythia8-config --cxxflags --ldflags used to build; see build below.
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/PowhegHooks.h"
#include <fstream>
#include <cmath>
using namespace Pythia8;

int main(int argc, char* argv[]) {
  if (argc < 4) { std::cerr << "usage: prog LHE NEV OUT.csv\n"; return 1; }
  std::string lhe = argv[1]; long nev = std::atol(argv[2]); std::string out = argv[3];

  Pythia pythia;
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe);
  // --- clean tune: Monash 2013 default, MPI on (no ad hoc overrides) ---
  pythia.readString("Tune:pp = 14");
  pythia.readString("PartonLevel:MPI = on");
  // --- STANDARD POWHEG matching (main164powheg.cmnd, HED=201) ---
  pythia.readString("POWHEG:veto = 1");
  pythia.readString("POWHEG:nFinal = -1");    // modern default: extract emission, no strict count
  pythia.readString("POWHEG:vetoCount = 3");
  pythia.readString("POWHEG:pThard = 0");     // required for nFinal<0: pThard = SCALUP
  pythia.readString("POWHEG:pTemt = 0");
  pythia.readString("POWHEG:pTdef = 1");
  pythia.readString("POWHEG:emitted = 0");
  pythia.readString("POWHEG:MPIveto = 0");
  pythia.readString("POWHEG:QEDveto = 2");
  // pTmaxMatch=2 + attach the real hook, exactly as main164 does when POWHEG:veto>0
  pythia.readString("SpaceShower:pTmaxMatch = 2");
  pythia.readString("TimeShower:pTmaxMatch  = 2");
  pythia.readString("Print:quiet = on");
  auto powhegHooks = make_shared<PowhegHooks>();
  pythia.setUserHooksPtr((UserHooksPtr)powhegHooks);
  if (!pythia.init()) { std::cerr << "init failed\n"; return 1; }

  std::ofstream f(out);
  static const char* WIDS[6] = {"1002","1003","1004","1005","1006","1007"};
  f << "qT,m,dphi,phistar,ptl0,ptl1,etal0,etal1,w,w1002,w1003,w1004,w1005,w1006,w1007\n";
  f.setf(std::ios::scientific); f.precision(7);
  long i = 0, ntry = 0;
  while (i < nev) {
    if (!pythia.next()) { if (pythia.info.atEndOfFile()) break; if (++ntry > 1000) break; continue; }
    // hardest l- (id>0) and l+ (id<0)
    int ia = -1, ib = -1; double pta = -1, ptb = -1;
    for (int k = 0; k < pythia.event.size(); ++k) {
      Particle& p = pythia.event[k];
      if (!p.isFinal()) continue;
      int id = p.id();
      if (id == 11 || id == 13) { if (p.pT() > pta) { pta = p.pT(); ia = k; } }
      if (id == -11 || id == -13) { if (p.pT() > ptb) { ptb = p.pT(); ib = k; } }
    }
    if (ia < 0 || ib < 0) continue;
    double L[2][4];
    for (int j = 0; j < 4; ++j) {
      L[0][j] = (j==0?pythia.event[ia].px():j==1?pythia.event[ia].py():j==2?pythia.event[ia].pz():pythia.event[ia].e());
      L[1][j] = (j==0?pythia.event[ib].px():j==1?pythia.event[ib].py():j==2?pythia.event[ib].pz():pythia.event[ib].e());
    }
    double ea = pythia.event[ia].eta(), pa = pythia.event[ia].phi();
    double eb = pythia.event[ib].eta(), pb = pythia.event[ib].phi();
    // DRESS: FSR photons within dR<0.1 of either lepton
    for (int k = 0; k < pythia.event.size(); ++k) {
      Particle& p = pythia.event[k];
      if (!p.isFinal() || p.id() != 22 || p.pT() < 1e-6) continue;
      double de0 = p.eta()-ea, dp0 = std::abs(p.phi()-pa); if (dp0>M_PI) dp0=2*M_PI-dp0;
      double de1 = p.eta()-eb, dp1 = std::abs(p.phi()-pb); if (dp1>M_PI) dp1=2*M_PI-dp1;
      double d0 = std::hypot(de0,dp0), d1 = std::hypot(de1,dp1);
      if (std::min(d0,d1) < 0.1) { int kk = d0<d1?0:1;
        L[kk][0]+=p.px(); L[kk][1]+=p.py(); L[kk][2]+=p.pz(); L[kk][3]+=p.e(); }
    }
    double px=L[0][0]+L[1][0], py=L[0][1]+L[1][1], pz=L[0][2]+L[1][2], e=L[0][3]+L[1][3];
    double m2 = e*e-px*px-py*py-pz*pz;
    if (m2 <= 1600.) continue;                       // m_ll > 40 GeV
    double p0 = std::hypot(L[0][0],L[0][1]), p1 = std::hypot(L[1][0],L[1][1]);
    double c = (L[0][0]*L[1][0]+L[0][1]*L[1][1]) / std::max(p0*p1,1e-300);
    c = std::min(1.,std::max(-1.,c));
    double dphi = std::acos(c);
    auto eta_of=[](double* v){ double pt=std::hypot(v[0],v[1]); return pt>1e-12?std::asinh(v[2]/pt):(v[2]>0?30.:-30.); };
    double em = eta_of(L[0]), ep = eta_of(L[1]);
    double cts = std::tanh(0.5*(em-ep));
    double phistar = std::tan(0.5*(M_PI-dphi)) * std::sqrt(std::max(0.,1.-cts*cts));
    double w = pythia.info.weight();
    f << std::hypot(px,py) << "," << std::sqrt(m2) << "," << dphi << "," << phistar << ","
      << p0 << "," << p1 << "," << em << "," << ep << "," << w;
    for (int q = 0; q < 6; ++q) f << "," << pythia.info.getWeightsDetailedValue(WIDS[q]);
    f << "\n";
    if (++i % 25000 == 0) { std::cerr << "  " << i << "/" << nev << "\n"; }
  }
  f.close();
  pythia.stat();
  std::cerr << "DONE " << i << " events -> " << out << "\n";
  return 0;
}
