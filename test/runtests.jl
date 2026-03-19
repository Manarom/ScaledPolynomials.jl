using ScaledPolynomials
using Test
using StaticArrays, Test, BenchmarkTools, LinearAlgebra, Test
using Polynomials
using SpecialPolynomials
using LegendrePolynomials


show_title(st) = begin 
    println("-------------------------------------")
    println(st)
    println("-------------------------------------")
end
show_subtitle(st,n = 1) = begin
    sh = mapreduce(_-> "==",*,1 : n)
    println("$sh > $(string(st))")
end

is_show_benchmarks = get(ENV, "CI" ,"false") == "false"

@testset "ScaledPolynomials" begin

        a = (0.1,-0.5,0.4,-1.0 , 20.0 , 45.6 , -1.5 , -100.0)
        N = length(a)
        show_title("Testing polynomials evaluation:")
        for x_i in (0.2 , 0.4 , -0.5 , 0.9)
            @test Polynomials.Polynomial(a)(x_i) ≈ ScaledPolynomials.StandPoly(a)(x_i)
            @test  Polynomials.ChebyshevT(a)(x_i) ≈ ScaledPolynomials.ChebPoly(a)(x_i)
            @test  LegendrePolynomials.Pl(x_i , 3) ≈ ScaledPolynomials.LegPoly((0.0, 0.0, 0.0, 1.0))(x_i)
            @test  LegendrePolynomials.Pl(x_i , 2) ≈ ScaledPolynomials.LegPoly((0.0, 0.0, 1.0, 0.0))(x_i)
            @test  LegendrePolynomials.Pl(x_i , 1) ≈ ScaledPolynomials.LegPoly((0.0, 1.0, 0.0, 0.0))(x_i)
            @test  LegendrePolynomials.Pl(x_i , 0) ≈ ScaledPolynomials.LegPoly((1.0, 0.0, 0.0, 0.0))(x_i)
            @test  SpecialPolynomials.Bernstein{N - 1}(a)(x_i) ≈ ScaledPolynomials.BernsteinPoly(a)(x_i)
        end    
        a = [1.0 , 22.0 , 3.0 , 4.56 , 3.51]
        N = length(a)
        x = range(-1.0,1.0,10)
        show_subtitle("Testing polynomial derivative coefficients and benchmarking the evaluation speed")
        for (TestPolyType,CheckPolyType) in zip((SpecialPolynomials.Bernstein{N - 1},
                                SpecialPolynomials.Chebyshev,
                                SpecialPolynomials.Legendre ), (ScaledPolynomials.BernsteinSymPoly, 
                            ScaledPolynomials.ChebPoly, ScaledPolynomials.LegPoly))
                show_subtitle("Testing  $(CheckPolyType)", 2)
                b_test = TestPolyType(a)
                b_der_test = Polynomials.derivative(b_test)
                b_pw = CheckPolyType(a)
                b_der = ScaledPolynomials.derivative(b_pw)
                if is_show_benchmarks
                    show_subtitle("$(TestPolyType)(x) evaluation: ", 3)
                    @btime $b_test(0.5)
                    show_subtitle(" $(CheckPolyType)(x) evaluation: ", 3)
                    @btime $b_pw(0.5)
                end
                show_subtitle("derivative( $(TestPolyType) ) evaluation: ",3)
                is_show_benchmarks && @btime $Polynomials.derivative($b_test)
                show_subtitle("derivative( $(CheckPolyType) ) evaluation: ",3)
                is_show_benchmarks && @btime $ScaledPolynomials.derivative($b_pw)

                if ~(TestPolyType <: SpecialPolynomials.Bernstein)
                    @test norm(ScaledPolynomials.coeffs(b_der) .- b_der_test.coeffs)≈ 0 atol = 1e-12
                else # BErnstein basis scaled from 0 to  1
                    @test norm(ScaledPolynomials.coeffs(b_der) .- b_der_test.coeffs./2)≈ 0 atol = 1e-12
                end

        end

        x = collect(range(-1.0,1.0,100))
        a = (1.0,22.0,3.0,4.56,3.51)
        show_title("Default scaling polynomial derivative evaluation test:")
        for CheckPolyType in (ScaledPolynomials.BernsteinSymPoly, ScaledPolynomials.StandPoly, 
                                    ScaledPolynomials.ChebPoly, ScaledPolynomials.LegPoly,
                                     ScaledPolynomials.TrigPoly)

            show_subtitle("Cheking the derivative for $CheckPolyType")
            pw_poly = CheckPolyType(a)
            pw_poly_vals = pw_poly.(x)
            is_trig = CheckPolyType <: ScaledPolynomials.TrigPoly

            s_fit = Polynomials.fit(Polynomials.Polynomial, x, pw_poly_vals, is_trig ? 20 : ScaledPolynomials.poly_degree(pw_poly)) 
    
            s_der = Polynomials.derivative(s_fit)
            pw_poly_der = ScaledPolynomials.derivative(pw_poly)
            @test norm(s_der.(x) .- pw_poly_der.(x)) ≈ 0 atol= is_trig ? 1e-10 : 1e-12 # need to chenge the accuracy for trig because of bad fitting
        end


        #Scaled polynomial derivatives
        x_unscaled = collect(200.0:0.1:2000.0)
        a = (1.0,22.0,3.0,4.56,3.51)
        (x_scaled,x_min,x_max) = ScaledPolynomials.scale_x_to_ξ(x_unscaled)

        p_scaled = ScaledPolynomials.ScaledPolynomial(ScaledPolynomials.ChebPoly, a,  x_min, x_max)
        y_values = p_scaled.(x_unscaled)
        p_scaled_der = ScaledPolynomials.derivative(p_scaled)
        pp = Polynomials.fit(Polynomials.Polynomial,x_unscaled, y_values, length(a) - 1)
        y_der_test = Polynomials.derivative(pp).(x_unscaled)

        show_title("Scaled polynomial derivative evaluation test ")
        for CheckPolyType in (ScaledPolynomials.BernsteinSymPoly, ScaledPolynomials.StandPoly, ScaledPolynomials.ChebPoly , ScaledPolynomials.LegPoly)
            show_subtitle("Cheking the derivative for $CheckPolyType")
            p_scaled = ScaledPolynomials.ScaledPolynomial(CheckPolyType, a,  x_min, x_max)
            y_values = p_scaled.(x_unscaled)
            @test norm(y_values .- CheckPolyType(a).(x_scaled)) ≈ 0.0 atol =1e-12
            # 
            pp = Polynomials.fit(Polynomials.Polynomial,x_unscaled, y_values, length(a) - 1)
            y_der_test = Polynomials.derivative(pp).(x_unscaled)

            p_scaled_der = ScaledPolynomials.derivative(p_scaled)

            @test norm( p_scaled_der.(x_unscaled) .- y_der_test ) ≈ 0.0 atol = 1e-12

        end
        println("-------------------------------------")
end
