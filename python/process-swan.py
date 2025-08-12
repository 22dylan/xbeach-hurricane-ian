import os



def process_swan_to_points(n_header, n_locs):
    fn_swan = os.path.join(os.getcwd(), "..", "data", "forcing", "spts01.out")
    header_lines = []
    swan_spectra = {i: [] for i in range(n_locs)}
    swan_loc_cnt = n_locs - 1
    with open(fn_swan,'r') as f:
        for cnt, line in enumerate(f.readlines()):
            if cnt < n_header:
                header_lines.append(line)
                if ("number of locations" in line):
                    n_locs_check = int([x.strip() for x in line.split()][0])
                    if n_locs != n_locs_check:
                        raise ValueError("Number of locations in input and swan file do not match. ")
            
            else:
                if ("date and time" in line):
                    curr_date_time = line
                    continue

                if ("ZERO" in line) or ("FACTOR" in line) or ("NODATA" in line):
                    swan_loc_cnt += 1
                    if swan_loc_cnt == 7:
                        swan_loc_cnt = 0
                    swan_spectra[swan_loc_cnt].append(curr_date_time)
                
                swan_spectra[swan_loc_cnt].append(line)


    for spectra in range(n_locs):
        fn_out = os.path.join(os.getcwd(), "test{}.out" .format(spectra+1))
        latlong_written = False
        with open(fn_out, 'w') as f:
            for l in header_lines:
                if ("number of locations" in l):
                    l = "1                                  number of locations\n"
                    f.write(l)
                    continue
                if "-81." in l:
                    if latlong_written == False:
                        f.write("  0   0\n")
                    latlong_written = True
                    continue

                f.write(l)
            
            for l in swan_spectra[spectra]:
                f.write(l)

if __name__=="__main__":
    process_swan_to_points(n_header=100, n_locs=7)



