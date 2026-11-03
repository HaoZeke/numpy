! Kinds that gfortran and LLVM flang select alike, which are those
! needing neither flang's real(2) and real(3) nor gfortran's real(16)
module selected_kinds
  implicit none
  integer, parameter :: srk_4_0 = selected_real_kind(4, 0), &
    srk_4_37 = selected_real_kind(4, 37), &
    srk_4_38 = selected_real_kind(4, 38), &
    srk_4_307 = selected_real_kind(4, 307), &
    srk_4_308 = selected_real_kind(4, 308), &
    srk_4_4931 = selected_real_kind(4, 4931), &
    srk_4_4932 = selected_real_kind(4, 4932)
  integer, parameter :: srk_6_0 = selected_real_kind(6, 0), &
    srk_6_37 = selected_real_kind(6, 37), &
    srk_6_38 = selected_real_kind(6, 38), &
    srk_6_307 = selected_real_kind(6, 307), &
    srk_6_308 = selected_real_kind(6, 308), &
    srk_6_4931 = selected_real_kind(6, 4931), &
    srk_6_4932 = selected_real_kind(6, 4932)
  integer, parameter :: srk_7_0 = selected_real_kind(7, 0), &
    srk_7_37 = selected_real_kind(7, 37), &
    srk_7_38 = selected_real_kind(7, 38), &
    srk_7_307 = selected_real_kind(7, 307), &
    srk_7_308 = selected_real_kind(7, 308), &
    srk_7_4931 = selected_real_kind(7, 4931), &
    srk_7_4932 = selected_real_kind(7, 4932)
  integer, parameter :: srk_15_0 = selected_real_kind(15, 0), &
    srk_15_37 = selected_real_kind(15, 37), &
    srk_15_38 = selected_real_kind(15, 38), &
    srk_15_307 = selected_real_kind(15, 307), &
    srk_15_308 = selected_real_kind(15, 308), &
    srk_15_4931 = selected_real_kind(15, 4931), &
    srk_15_4932 = selected_real_kind(15, 4932)
  integer, parameter :: srk_16_0 = selected_real_kind(16, 0), &
    srk_16_37 = selected_real_kind(16, 37), &
    srk_16_38 = selected_real_kind(16, 38), &
    srk_16_307 = selected_real_kind(16, 307), &
    srk_16_308 = selected_real_kind(16, 308), &
    srk_16_4931 = selected_real_kind(16, 4931), &
    srk_16_4932 = selected_real_kind(16, 4932)
  integer, parameter :: srk_18_0 = selected_real_kind(18, 0), &
    srk_18_37 = selected_real_kind(18, 37), &
    srk_18_38 = selected_real_kind(18, 38), &
    srk_18_307 = selected_real_kind(18, 307), &
    srk_18_308 = selected_real_kind(18, 308), &
    srk_18_4931 = selected_real_kind(18, 4931), &
    srk_18_4932 = selected_real_kind(18, 4932)
  integer, parameter :: srk_34_0 = selected_real_kind(34, 0), &
    srk_34_37 = selected_real_kind(34, 37), &
    srk_34_38 = selected_real_kind(34, 38), &
    srk_34_307 = selected_real_kind(34, 307), &
    srk_34_308 = selected_real_kind(34, 308), &
    srk_34_4931 = selected_real_kind(34, 4931), &
    srk_34_4932 = selected_real_kind(34, 4932)
  integer, parameter :: srk_40_0 = selected_real_kind(40, 0), &
    srk_40_37 = selected_real_kind(40, 37), &
    srk_40_38 = selected_real_kind(40, 38), &
    srk_40_307 = selected_real_kind(40, 307), &
    srk_40_308 = selected_real_kind(40, 308), &
    srk_40_4931 = selected_real_kind(40, 4931), &
    srk_40_4932 = selected_real_kind(40, 4932)
  integer, parameter :: srk_r400 = selected_real_kind(r=400), &
    srk_r5000 = selected_real_kind(r=5000), &
    srk_radix10 = selected_real_kind(radix=10), &
    srk_6_radix10 = selected_real_kind(6, radix=10), &
    srk_7_0_2 = selected_real_kind(7, 0, 2)
  integer, parameter :: sik_m1 = selected_int_kind(-1), &
    sik_0 = selected_int_kind(0), &
    sik_2 = selected_int_kind(2), &
    sik_3 = selected_int_kind(3), &
    sik_4 = selected_int_kind(4), &
    sik_5 = selected_int_kind(5), &
    sik_9 = selected_int_kind(9), &
    sik_10 = selected_int_kind(10), &
    sik_18 = selected_int_kind(18), &
    sik_19 = selected_int_kind(19), &
    sik_38 = selected_int_kind(38), &
    sik_39 = selected_int_kind(39)
  integer, parameter :: sck_default = selected_char_kind('default'), &
    sck_ascii = selected_char_kind('ASCII '), &
    sck_ucs4 = selected_char_kind('iso_10646'), &
    sck_utf8 = selected_char_kind('utf-8')
end module selected_kinds
