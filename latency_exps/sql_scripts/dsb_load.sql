\copy date_dim from '/home/ubuntu/dsb/data/date_dim.dat' with csv delimiter '|' quote '"' escape '\';
\copy customer_demographics from '/home/ubuntu/dsb/data/customer_demographics.dat' with csv delimiter '|' quote '"' escape '\';
\copy household_demographics from '/home/ubuntu/dsb/household_demographics.dat' with csv delimiter '|' quote '"' escape '\';
\copy store from '/home/ubuntu/dsb/data/store.dat' with csv delimiter '|' quote '"' escape '\';
\copy store_sales from '/home/ubuntu/dsb/data/store_sales.dat' with csv delimiter '|' quote '"' escape '\';
