SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info mi, movie_info_idx mi_idx, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<=106 AND ci.role_id>2 AND t.kind_id=6 AND t.production_year>=2002 AND mc.company_type_id<=1;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info mi, movie_info_idx mi_idx, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>1999 AND mi.info_type_id=16 AND mc.company_id>12063 AND ci.role_id>3 AND mi_idx.info_type_id<101;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_type_id>=2 AND t.production_year<1997;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id<=7 AND mc.company_id=20510 AND t.production_year>1991 AND t.kind_id>=1;
SELECT COUNT(*) FROM cast_info ci, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND t.production_year<=1971 AND ci.role_id<10;
SELECT COUNT(*) FROM cast_info ci, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2009 AND mi_idx.info_type_id>=99 AND mk.keyword_id<43;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>4976 AND mc.company_type_id<=1 AND ci.role_id=4 AND t.production_year=1994 AND t.kind_id<6;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>10700 AND t.kind_id<=1 AND t.production_year=2009 AND mi.info_type_id=17 AND ci.role_id<10 AND mc.company_type_id<=2 AND mk.keyword_id=8449;
SELECT COUNT(*) FROM movie_companies mc, movie_keyword mk, title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.production_year<1896;
SELECT COUNT(*) FROM movie_companies mc, movie_keyword mk, title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id<=7 AND mc.company_type_id>1 AND t.production_year<=1995;
SELECT COUNT(*) FROM cast_info ci, movie_info_idx mi_idx, title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id>1 AND ci.role_id>=2 AND t.production_year=2006;
SELECT COUNT(*) FROM cast_info ci, movie_info_idx mi_idx, title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>1 AND ci.role_id=5 AND t.production_year>2007;
SELECT COUNT(*) FROM movie_companies mc, movie_info mi, movie_info_idx mi_idx, title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<=16 AND t.production_year=2011 AND mi_idx.info_type_id<101 AND mc.company_id<136933 AND mc.company_type_id<=1 AND t.kind_id>=1;
SELECT COUNT(*) FROM movie_companies mc, movie_info mi, movie_info_idx mi_idx, title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>=2013 AND mc.company_id>84346 AND mc.company_type_id>1;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info_idx mi_idx, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id>1 AND mi_idx.info_type_id>100 AND t.production_year>1978;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info_idx mi_idx, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND ci.role_id>=10 AND mc.company_id<=121941 AND t.kind_id>=1 AND mc.company_type_id>=1 AND t.production_year>=2005;
SELECT COUNT(*) FROM movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<53165 AND t.production_year>2009;
SELECT COUNT(*) FROM movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=6 AND t.production_year>=2008 AND mk.keyword_id>8580 AND mi.info_type_id>=15 AND mi_idx.info_type_id>100;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info mi, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>=1922 AND mc.company_type_id>=1 AND t.kind_id=3;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info mi, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mc.company_id>=10277 AND ci.role_id>5 AND mi.info_type_id<13 AND mk.keyword_id>=57655 AND t.kind_id=1 AND mc.company_type_id>=2 AND t.production_year<1907;
SELECT COUNT(*) FROM movie_companies mc, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>11474 AND mk.keyword_id>1962 AND t.kind_id>3 AND t.production_year=2009 AND mc.company_type_id>=1;
SELECT COUNT(*) FROM movie_companies mc, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id<=146828 AND t.production_year>=2012 AND mk.keyword_id>118001;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id<475 AND ci.role_id<=4 AND t.production_year=2004 AND t.kind_id<=1 AND mk.keyword_id>745 AND mc.company_type_id=1;
SELECT COUNT(*) FROM cast_info ci, movie_companies mc, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>521 AND mk.keyword_id>870 AND mi_idx.info_type_id>99 AND ci.role_id<=10 AND t.production_year=1903 AND mc.company_type_id=1;
SELECT COUNT(*) FROM movie_info mi, movie_keyword mk, title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=1 AND mk.keyword_id>=644 AND t.production_year=1972 AND mi.info_type_id<92;
SELECT COUNT(*) FROM movie_info mi, movie_keyword mk, title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>=11 AND t.production_year>=1994;
SELECT COUNT(*) FROM movie_companies mc, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<112 AND t.production_year>2006 AND mc.company_type_id>1 AND mi.info_type_id<=96;
SELECT COUNT(*) FROM movie_companies mc, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<=89857 AND mc.company_id<5286 AND mi.info_type_id>=13 AND t.production_year=2006;
SELECT COUNT(*) FROM movie_companies mc, movie_info mi, title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<2021 AND t.production_year<1990 AND mi.info_type_id>=5 AND mc.company_type_id=1 AND t.kind_id=1;
SELECT COUNT(*) FROM movie_companies mc, movie_info mi, title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<686 AND t.kind_id=1 AND mc.company_type_id>=1 AND t.production_year>1915 AND mi.info_type_id<15;