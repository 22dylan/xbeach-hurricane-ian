import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class plot_forcing():
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_to_forcing =  os.path.join(self.file_dir, "..", "..", "..", "data", "forcing", "xbeach4.dat")

        
    def plot(self, var="el", savefig=False):
        label, ylabel, color = self.var2label(var)
        df = self.frcing_to_dataframe(self.path_to_forcing)
        
        stop_idx = df.loc[df["t_sec"]==150300].index[0]
        df_trnc = df.iloc[0:stop_idx]

        fig, ax = plt.subplots(1,1, figsize=(4,3))
        ax.plot(df["t_hr"], df[var], color="k", lw=1.5, label=label)
        # ax.plot(df_trnc["t_hr"], df_trnc[var], color="#ff5370", lw=3, label="Truncated Elevation")

        # ax.plot(df["t_sec"], df["el"], color='dodgerblue', lw=1.5, label="Water Elevation")
        ax.legend()
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel(ylabel)

        if savefig:
            fn = "{}.png" .format(var)
            plt.savefig(fn,
                        transparent=False, 
                        dpi=500,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()



    def var2label(self, var):
        v2l = { "el":"Water Elevation",
                "hs": "Significant Wave Height",
                "Tp": "Peak Period",
                "wavedir": "Wave Direction"
        }
        v2y = { "el": "Water Elevation (m; ___)",
                "hs": "Significant Wave Height (m)",
                "Tp": "Peak Period (s)",
                "wavedir": "Wave Direction"
        }
        
        c = {"el": 0, "hs": 1, "Tp": 2, "wavedir": 3}
        colors = sns.color_palette("crest", n_colors=len(c.keys()))
        color = colors[c[var]]

        return v2l[var], v2y[var], color

    def frcing_to_dataframe(self, fn, n_header=3, n_var=7):
        t, el, wx, wy, hs, Tp, wavedir = [], [], [], [], [], [], [],
        with open(self.path_to_forcing,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if cnt < n_header:
                    if "VARIABLES" in line:
                        var = [x.strip() for x in line.split()]
                        var = [i for i in var if i!="VARIABLES"]
                        var = [i for i in var if i!="="]
                    continue
                t_, el_, wx_, wy_, hs_, Tp_, wavedir_ = [float(x.strip()) for x in line.split()]
                t.append(t_)
                el.append(el_)
                wx.append(wx_)
                wy.append(wy_)
                hs.append(hs_)
                Tp.append(Tp_)

                # wavedir_ = self.cartesian_to_nautical_angle(wavedir_)
                # wavedir_ = self.nautical_to_xbeach_angle(wavedir_, alfa=55.92839019260679)

                wavedir.append(wavedir_)
        

        # TODO confirm unit conversions with Don
        df = pd.DataFrame()
        df["t"] = t
        df["el"] = el
        df["wx"] = wx
        df["wy"] = wy
        df["hs"] = hs
        df["Tp"] = Tp
        df["wavedir"] = wavedir

        df["el"] = df["el"]*0.3048
        df["hs"] = df["hs"]*0.3048

        dt = (df["t"].iloc[1] - df["t"].iloc[0])*60*60         # tiime setp in seconds; converting from hours.
        df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))
        df["t_hr"] = df["t_sec"]/3600

        return df
        

    def cartesian_to_nautical_angle(self, deg):
        """ converting from cartesian to nautical angles for xbeach input
        Cartesian: waves traveling TO east are zero and counterclockwise is positive.
        Nautical: waves traveling FROM North are zero and clockwise is positive. 
        """
        if (deg>=0) & (deg <= 270):
           return (270-deg)
        elif (deg>270) & (deg<360):
           return (270-deg)+360
        else:
            raise ValueError("{} must be between 0 and 360." .format(deg))

    def nautical_to_xbeach_angle(self, deg, alfa):
        """
        """
        deg = deg + alfa 
        if deg > 360:
            deg -= 360
        elif deg < 0:
            deg += 360
        return deg


if __name__ == "__main__":
    pf = plot_forcing()
    pf.plot(var="wavedir", savefig=False)
    # pf.plot(var="hs", savefig=False)
    # pf.plot(var="Tp", savefig=False)


    plt.show()