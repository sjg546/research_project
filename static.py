from pdf import PDF
GMID = 0

def Vwin(t:float, e:float):
   PDF.pdf(t - e) / cdf(t - e);


static double Wwin(double t, double e) {
  double vwin = Vwin(t, e);
  return vwin * (vwin + t - e);
}

static double Vdraw(double t, double e) {
  return (pdf(-e - t) - pdf(e - t)) / (cdf(e - t) - cdf(-e - t));
}

static double Wdraw(double t, double e) {
  double vdraw = Vdraw(t, e);
  double n = (vdraw * vdraw) + ((e - t) * pdf(e - t) + (e + t) * pdf(e + t));
  double d = cdf(e - t) - cdf(-e - t);
  return n / d;
}
