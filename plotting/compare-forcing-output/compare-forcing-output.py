


class compare_forcing_output():
    """docstring for xb_plotting_pt"""
    def __init__(self, model_runname, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_runname = model_runname
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models", self.model_runname)
        self.var = var
        self.xgr, self.ygr = self.read_grid()
    
    def read_data_xarray(self, var, t, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        slice_data = ds[var].isel(globaltime=slice(t,t+1))
        if rtn_time_array:
            time = ds["globaltime"].values
            # print("Last time step: {} hr." .format(time[-1]/60/60))
            return slice_data.values[0,:,:], time
        else:
            return slice_data.values[0,:,:]

    def read_grid(self):
        fn = os.path.join(self.path_to_model, "x.grd")
        if os.path.isfile(fn):
            xgrid = os.path.join(self.path_to_model, "x.grd")
            ygrid = os.path.join(self.path_to_model, "y.grd")

            with open(xgrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    xs = [float(i.strip()) for i in line.split()]
                    if cnt == 0:
                        break
                    
            ys = []
            with open(ygrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    y_ = [float(i.strip()) for i in line.split()][0]
                    ys.append(y_)


            xgr, ygr = np.meshgrid(xs, ys)
        else:
            fn_params = os.path.join(self.path_to_model, "params.txt")
            with open(fn_params) as f:
                for cnt, line in enumerate(f.readlines()):
                    ls = [i.strip() for i in line.split()]
                    if "dx" in ls:
                        dx = float(ls[-1])
                    elif "dy" in ls:
                        dy = float(ls[-1])
                    elif "nx" in ls:
                        nx = float(ls[-1])
                    elif "ny" in ls:
                        ny = float(ls[-1])
            
            xs = np.arange(start=0, stop=nx*dx+dx, step=dx)
            ys = np.arange(start=0, stop=ny*dy+dx, step=dy)
            xgr, ygr = np.meshgrid(xs, ys)

        return xgr, ygr

     def compare_forcing2output(self, xb_loc, drawdomain=False, savefig=False):
        if xb_loc == "lower-left":
            xidx = 1
            yidx = 1
            frcng_dat = "xbeach1.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"
                    
        elif xb_loc == "upper-left":
            xidx = 1
            yidx = np.shape(self.data)[1]-1
            frcng_dat = "xbeach4.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"
        
        elif xb_loc == "upper-right":
            xidx = np.shape(self.data)[2]-1
            yidx = np.shape(self.data)[1]-1
            frcng_dat = "xbeach3.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"

        elif xb_loc == "lower-right":
            xidx = np.shape(self.data)[2]-1
            yidx = 0
            frcng_dat = "xbeach2.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"

        elif xb_loc == "adcirc-offshore":
            xidx = np.argmin(np.abs(self.xgr[0,:] - 2585))
            yidx = np.argmin(np.abs(self.ygr[:,0] - 5640))
            frcng_dat = "xbeach5.dat"
            xb_label = "XBeach"
            fr_label = "ADCIRC/SWAN"

        elif xb_loc == "adcirc-onshore":
            xidx = np.argmin(np.abs(self.xgr[0,:] - 3069))
            yidx = np.argmin(np.abs(self.ygr[:,0] - 5625))
            frcng_dat = "xbeach6.dat"
            xb_label = "XBeach"
            fr_label = "ADCIRC/SWAN"

        else:
            raise ValueError("Incorrect xb_loc provided.")    

        output = self.data[:,yidx, xidx]
        output[output<-99999] = 0

        fn = os.path.join(self.file_dir, "..", "..", "..", "data", "forcing", frcng_dat)
        df = self.frcing_to_dataframe(fn)

        fig, ax = plt.subplots(1,1, figsize=(8,3))
        ax.plot(self.time/3600, output, color="dodgerblue", lw=2.5, label=xb_label)
        ax.plot(df["t_sec"]/3600, df["el"], color="k", lw=1.5, label=fr_label)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Water Elevation (m)")
        ax.set_xlim([0,45])
        ax.legend()

        if savefig == True:
            fn = "frcng2output.png"
            plt.savefig(fn,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


        if drawdomain:
            data_plot = self.data[0,:,:]
            fig, ax = plt.subplots(1,1, figsize=(4,8))
            # --- new
            mask = (data_plot < -99999)
            masked_array = np.ma.array(data_plot, mask=mask)
            cmap = mpl.cm.Blues
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-0.5, vmax=np.max(self.data), cmap=cmap)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            
            cnt = 0
            colors = sns.color_palette("husl")
            x, y = self.xgr[0,xidx], self.ygr[yidx, 0]                
            ax.scatter(x, y, color=colors[cnt],s=50)
            ax.annotate("{}" .format(cnt), (x, y))

        if savefig == True:
            fn = "frcng2output_domain.png"
            plt.savefig(fn,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


    def frcing_to_dataframe(self, fn, n_header=3):
        t, el, wx, wy, hs, Tp, wavedir = [], [], [], [], [], [], [],
        with open(fn,'r') as f:
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

        dt = (df["t"].iloc[1] - df["t"].iloc[0])*60*60         # time setp in seconds; converting from hours.
        df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))
        df["t_hr"] = df["t_sec"]/3600

        return df

if __name__ == "__main__":
    cfo = compare_forcing_output("gvm-run3-30m-nobldgs", var="H")
    cfo.compare_forcing2output(xb_loc="lower-left")


    