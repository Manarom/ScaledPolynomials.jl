### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d20e7502-fa03-4448-84f4-dc46acf52c9b
try 
	using ScaledPolynomials
	using Plots,StaticArrays,Interpolations,  PlutoUI, LinearAlgebra, PrettyTables, Optimization,OptimizationOptimJ
catch notfound 
	import Pkg 
	Pkg.activate(@__DIR__())
	Pkg.instantiate()
	using ScaledPolynomials
	using Plots,StaticArrays,Interpolations,  PlutoUI, LinearAlgebra, PrettyTables, Optimization,OptimizationOptimJL
end

# ╔═╡ 14fe8399-7d59-4382-bc53-6e9ae4c05a8c
PlutoUI.TableOfContents(indent=true, depth=4, aside=true)

# ╔═╡ f240500d-1198-49f7-b043-beea86a248c7
md"""
## Bernstein polynomial for constraint optimization

#### Introduction

A common problem in constrained nonlinear optimization is converting nonlinear constraints applied to some function of the optimization variables into box constraints on the optimization variables themselves. Simply, converting nonlinear constraints to box constraints is necessary.

For example, in multiwavelength pyrometry, the measured signal can be approximated as a product of two functions: one (spectral emissivity) is linear with respect to the optimization variables, while the other (`blackbody` thermal emission spectrum) is nonlinear.

`` \vec{b}^*=argmin\{F(\vec{b})\}`` - optimization problem

``F(\vec{b})=\sum_{i=1}^{M}[y_i - I_{bb}(\lambda_i,T) \cdot (\sum_{k=0}^{n}a_k \cdot \phi_k(λ_i))]^2``  - discrepancy function

``\vec{b} = [\vec{a},T]^t`` - optimization variables vector, ``T`` is the temperature,  ``\vec{a}`` - coefficients of emissivity approximation, ``I_{bb}(\lambda_i,T)`` - blackbody thermal emission spectrum, ``y_i``  - measured intensity.

``\epsilon(λ)=\sum_{n=0}^{N-1}a_n \cdot \phi_n(λ)`` - spectral emissivity is a linear combination of some basis functions ``\phi_n(λ)``

There is a physical constraint on the emissivity, which follows from the fact that real object cannot emit more radiation than the blackbody:

``\epsilon(\lambda) \in (0...1)`` for the whole spectral range.

Sometimes, the region of emissivity variation can be narrowed; for example, it may be known that, for a particular material, the emissivity lies within the range of ``[\epsilon_a ... \epsilon_b]``. The question is how to convert the emissivity variation region to the emissivity approximation coefficients ``\vec{a}`` variation.

System of inequalities of contraints on emissivity:  

``\vec{\epsilon_a} \leq \vec{\epsilon} \leq \vec{\epsilon_b}`` (vectors in independent variables space ``R^M``)

Should be somehow converted to inequality constraints on emissivity approximation variables:

``\vec{a_a} \leq \vec{a} \leq \vec{a_b}`` (vector in optimization variables space ``R^N`` !)

This problem can be solved using the Bernstein polynomial basis.

\
"""

# ╔═╡ c8102be0-6f8f-406e-9f18-45f462116602
md"""
#### Matrix form of polynomial approximation

Each polynomial basis has a set of basis functions (monomials):

``\begin{bmatrix} \phi_0(x) , \dots,  \phi_n(x) \end{bmatrix}`` 

E.g. for standard polynomial basis:  ``\phi_k = x^{k}``

Columns of Vandermonde matrix (``V``) are the polynomial basis functions evaluated at coordinates ``x``, thus number of columns in ``V`` is equal to the `degree of polynomial + 1`: 

``V= \begin{bmatrix} \vec{\phi_0} , \dots,  \vec{\phi_n} \end{bmatrix}`` - Vandermonde matrix

\
"""

# ╔═╡ 381fd867-f404-40e5-a270-ae98f50bf9b5
md"Select polynomial basis type for the initial function $(@bind poly_type Select(collect(keys( ScaledPolynomials.SUPPORTED_POLYNOMIAL_TYPES)),default = :stand))"

# ╔═╡ 1b15f3f8-fd8e-43c5-8c21-0b6fca139427
md"Initial function polynomial degree $(@bind polydegree Select(0:30,default = 3))"

# ╔═╡ 81edb295-c20f-47a9-8a6f-d21e475df6a9
begin 
	NPOINTS = 80
	x = SVector{NPOINTS}(collect(range(-1,1,NPOINTS)));
	PolyType =  ScaledPolynomials.SUPPORTED_POLYNOMIAL_TYPES[poly_type]{polydegree + 1,Float64}
	V =  ScaledPolynomials.VanderMatrix(x,PolyType())
	Vbern =  ScaledPolynomials.VanderMatrix(x,ScaledPolynomials.BernsteinSymPoly{polydegree + 1,Float64}())
end

# ╔═╡ 0dbf180e-8b27-4be7-8747-e424f37e2e50
md" Show infilled ? $(@bind isinfilled CheckBox(default = true))"

# ╔═╡ a6381315-af21-4117-beed-6e3f7b8b8ca9
md"""
The following figure shows basis functions (Vandermonde matrix columns) for polynomial of type **$(poly_type)**
"""

# ╔═╡ c6dfdffe-8dd7-4264-9b75-c435c5bbf941
plot(V,infill= isinfilled, title = "Basis functions for $(poly_type) polynomial bassis")

# ╔═╡ e243649b-b920-4ca3-b2f3-6e9b5ada13c9
md"""
### Bernstein polynomial basis

Bernstein polynomial basis of degree ``n`` has the following set of ``n + 1`` basis functions (monomials):

``\beta_{k}^{n}(x) = (\begin{matrix} n \\ k \end{matrix}) x^k(1-x)^{n-k}`` - Bernstein basis function for ``x \in [0...1]`` , ``k \in [0...n]``.

``(\begin{matrix} n \\ k \end{matrix}) = \frac {n!}{(n - k)!k!}``
"""

# ╔═╡ 7d6bf89c-4a52-4ed3-93ba-08c9f972c7a4
md"""
For general case, when ``x \in [a,b]``

``\beta_{k}^{n}(x) = (\begin{matrix} n \\ k \end{matrix}) (\frac {x - a}{b - a})^k(\frac {b-x}{b-a})^{n-k}`` - Bernstein basis function for ``x \in [a...b]`` , ``k \in [0...n]`` 

In the following, all formulas will be provided for ``x \in [0...1]`` basis, because of simplicity, but for calculations, symmetric Bernstein (**bernsteinsym** in selection dropdown) basis (``x \in [-1...1]``) was used because of the ransge ``[-1...1]`` is more natural for other polynomial bases, like Legendre and Chebyshev polynomials
"""

# ╔═╡ d130ca1e-94be-48b3-9c09-956a7a6b6bf4
md"""
##### 2. Each B-monomial has a single maximum with the value and location governed by ``n`` and ``k``.

B-basis function ``\beta_{k}^{n}(x)``  maximum value is ``max(\beta_{k}^{n}(x)) = k^k \cdot n^n \cdot (n-k)^{n-k} \cdot (\begin{matrix} n \\ k \end{matrix})``. It is located  at coordinate:  ``argmax(\beta_{k}^{n}(x)) = \frac{k}{n}``

"""

# ╔═╡ e9e28b59-36a2-406c-8f1e-488140aa9bc8
md"""

##### 3. All B-monomials are positive for all ``x``

"""

# ╔═╡ 9aad3160-fb56-4407-bdb7-f969d2b982a3
md"""

##### 4. The summation over all B-monomials within any B-basis set  gives one for any coordinate ``x``: 
``\Sigma_{k=0}^{n} \beta_{k}^{n}(x) = 1``. 

"""

# ╔═╡ 54e44992-423e-4ebb-92e6-2af89b52e7f7
md"The third property means that each row of **B-basis** Vandermonde matrix sums to one: ``\Sigma_i V[i,:] = 1``, hence ``\Sigma_{i,j} V[i,j] = M`` ,  M is the number of rows (size of x vector)"

# ╔═╡ bb55121f-9e67-446d-bf4f-4fc3dfe48062
md" ``\Sigma_{i,j} V[i,j] =`` $(sum(Vbern.v)) 

Is the summation of B Vandermonde matrix over all values for x with $(length(x))  points (equals to the number of x - points)"

# ╔═╡ 3280488d-efef-4833-ae37-6ea534f50df4
plot([sum(c) for c in  eachcol(Vbern.v)],label=nothing, marker = :diamond, title = " Σ of V columns vs column index")

# ╔═╡ facb5dae-3ee9-4380-8c92-0e94c93c84bc
md"""
Now, consider the expantion of function ``f(x)``  in **B-basis** of degree ``n``:

``f(x) = \Sigma_{k=0}^{n} a_k \beta_k^n(x)``

``\vec{f} = V \vec{a}`` - here ``\vec{f}`` is the vector of function values evaluated at ``\vec{x}`` coordinates.
(the least-square solution of this systemof equations is ``\vec{a} = V^{\dagger}\vec{f}``, where ``V^{\dagger}`` is pseudo-inverse)
"""

# ╔═╡ 14ef768c-7039-4f34-862d-f58acf89e7d6
md"""
Bernstein polynomial approximation coefficients:

"""

# ╔═╡ 56554cd8-e858-43d8-b6ef-7593d044920c

@bind  a PlutoUI.combine() do Child
	md"""
	Coefficients of f(x) function, ``f(x) = \Sigma_{k=0}^{n} a_k\phi_k^n``, where n=$(polydegree) and ``\phi_k^n`` are the basis functions for $(poly_type) basis
	
	``a_0`` = $(
		Child(Slider(-1:0.001:1,default=0.3,show_value = true))
	) \
	``a_1`` = $(
		Child(Slider(-1:0.001:1,default=0.1,show_value = true))
	)\
	``a_2`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_3`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_4`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_5`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_6`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_7`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_8`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_9`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	``a_{10}`` = $(
		Child(Slider(-1:0.001:1,default=0.2,show_value = true))
	)\
	"""
end

# ╔═╡ 3b475ddd-be05-4c56-9b28-68a808fc6e11
a_real =SVector{polydegree + 1}(a[1:polydegree + 1]);

# ╔═╡ 132a1594-326c-4d48-88ce-b7756656b606
md" Show B-basis fitting $(@bind is_show_bern_coeffs CheckBox(true))"

# ╔═╡ 7e1bcb20-a71a-4e9b-b797-dea6922f4cf8
md" Fix axes limits $(@bind fix_limits CheckBox(false))"

# ╔═╡ edeab52f-317e-42dc-840e-5bedece45dd8
@bind  lim_vals PlutoUI.combine() do Child
md"""
	
``y_{left}`` = $(Child(Slider(-5:1e-2:5,default = 0.0,show_value=true))) 	
``y_{right}`` = $( Child(Slider(-5:1e-2:5,default = 1.0,show_value = true)))
	
"""
end

# ╔═╡ 8c2bdb7c-aad4-4dd7-b6e2-97cf7dce0e56
md"Add noise $(@bind noise_amplitude Slider(0.0:1e-2:1, default =  0.0, show_value = true))"

# ╔═╡ faaa4542-2066-49fc-b978-a6a45a4cca4e
begin 
	y_bern = (1.0 .+ noise_amplitude*rand(NPOINTS)).*(V*a_real)
	sum(eachcol(V.v_unscaled)) # checking normalization
end;

# ╔═╡ 67918393-0efb-4c8a-a291-3d86f7853cce
md"""
##### Figure shows the following plots: 
1. Initial function ``f(x)`` (which is summation of monomials of basis $poly_type of degree $polydegree)
2. The result of ``f(x)`` fitting using B-polynomial basis: ``\Sigma_{k=0}^na_k\beta_k^n``
3. Scattered points are the values of ``a_k`` vs the locations of corresponding monomials ``\beta_k^n``
"""

# ╔═╡ fd446c33-d439-439f-bd0a-f36770856cd2
md" Bernstein fitting basis degree $(@bind bern_degree Select(1:50,default = 10))"

# ╔═╡ eb75c9bd-d3f1-4cd2-bcf1-b53e6f7bed96
md"""
The following table and figure show  $(@bind sub_monomial_degree Select(0:bern_degree))'th B-monomial for B-basis of degree $(bern_degree) fitting using standard basis polynomial
"""

# ╔═╡ 52c7978a-c51b-40ac-9604-0485ff469141
md"""

### Main features of B-polynomial basis:

##### 1. Each B-monomial ``\beta_{k}^{n}(x)`` is a polynomial of degree ``n``. 

Roughly speaking, all B-monomial have the same `units`

The following table shows the coeffieicnts of bernstein monomial fitting with standard basis for B-basis of degree  $(sub_monomial_degree)

"""

# ╔═╡ 1fa31f4b-7ed5-46fa-87ad-e5a4d22d59d6
begin 
	stand_bas = ScaledPolynomials.StandPoly{bern_degree + 1, Float64}
	bern_bas = ScaledPolynomials.BernsteinSymPoly{bern_degree + 1, Float64}()
	VV = ScaledPolynomials.vander(bern_bas , Vector(x))
	standart_basis_bern_monomial_fit_poly = ScaledPolynomials.polyfit(stand_bas,Vector(x),VV[: , sub_monomial_degree + 1])
	
	plot(x,VV[: , sub_monomial_degree + 1],label="Bernstein monomial $(sub_monomial_degree)")
	plot!(x,standart_basis_bern_monomial_fit_poly.(x),label = "Standard basis fit")
end

# ╔═╡ bf5df253-2d64-470b-8592-e93a6bcd144a
pretty_table(HTML,transpose(standart_basis_bern_monomial_fit_poly.coeffs) , top_left_string = "Standard basis polynomial fitting",column_labels = ["a_$(i)" for i in 0:ScaledPolynomials.parnumber(standart_basis_bern_monomial_fit_poly) - 1])

# ╔═╡ 67ec5900-5ab4-4b8d-b9ac-7d34a01a7229
begin # bernsteinfit block
	Bern_Type =  ScaledPolynomials.BernsteinSymPoly{bern_degree + 1,Float64}
	#bern_vander_mat = ScaledPolynomials.VanderMatrix(x,Bern_Type())
	bern_fit_poly = ScaledPolynomials.polyfit(Bern_Type,Vector(x),Vector(y_bern))
	ber_max_ocations = ScaledPolynomials.bern_max_locations(bern_fit_poly)
	bern_coeffs = bern_fit_poly.coeffs
	bern_fit = bern_fit_poly.(Vector(x))
end;

# ╔═╡ 0b00c857-aeb8-47ee-a9c2-b94fc32e9fb3
begin 
	function_values_at_bernstein_location = linear_interpolation(x,y_bern)(ber_max_ocations)
	delta = function_values_at_bernstein_location - bern_coeffs
	pretty_table(HTML,hcat(ber_max_ocations,bern_coeffs,function_values_at_bernstein_location,100*delta./function_values_at_bernstein_location) ,column_labels = ["Coordinate", "Bernstein coefficient", "Function value", "delta/F, %"], title = "Bernstein polynomial fitting")
end

# ╔═╡ c0418e2e-a6b6-4af5-a639-6d13952c88de
begin 
	
	ppp = plot(x,y_bern,label = "initial function f(x): $(poly_type) basis degree $(polydegree)")
	!fix_limits || ylims!(ppp,lim_vals)
	
	!is_show_bern_coeffs || begin 
		#plot!(ppp,x,fit_res[2], label = "B-basis fit Σbᵢβᵢ" )
		scatter!(ppp,ber_max_ocations,bern_coeffs, label = " aᵢ values located at B-monomials maxima")
	end
	
	ppp
end

# ╔═╡ 4349710e-529b-4036-8986-6140292457c2
md"""
##### It is clearly visible that the values of B-polynomials coefficients tend to the values of function itself!
-----------------------
The oposite is also true, if the initial function is a linear cobination of B-monomials, than the value of function in the vicinity of the corresponding monomial maxima location can be adjusted by adjusting the value of the coefficient (to check - set the polynomial bassis type of the initial function to **bernsteinsym** and use sliders)
"""

# ╔═╡ 9c73d7b0-547a-4577-8655-45a4e13df7a8
md"""

### Bernstein polynomials for constraints mapping

Now, we are going to the main topic - converting a nonlinear constraints to the box-constraints.

The problem is the following:

Fit the polynomial function ``f(\vec{a}, x)``  to the measured data ``y(x)`` with the following constraints:  ``f(\vec{a}, x) \le q(x)`` and ``f(\vec{a},x) \ge g(x)``, where ``\vec{a}`` is the the optimization variables vector, ``x`` is the independent variable,  ``g(x)`` and ``q(x)`` are known nonlinear functions. 

The loss function will be a least-square discrepancy:

``\Phi(\vec{a}) = \Sigma_i (y(x_i) - f(\vec{a},x_i))^2`` - loss function 

In mathematical form this gives:

``\vec{a^*} = argmin(\Phi(\vec{a}))``

Where the feasible region of ``\vec{a}`` is a subject to the following constraints:

``f(\vec{a}, x) \le q(x)``

``f(\vec{a},x) \ge g(x)``

"""

# ╔═╡ 7dfbac39-8e1b-4d88-830f-d18d94ed5e1b
md"""
We will be looking for the function ``f(\vec{a}, x)`` as an expantion in Bernstein basis:

``f(\vec{a},x) = \Sigma_j a_j\beta_j^n`` or in matrix form:  
``\vec{f} = V_{\beta}\cdot \vec{a}``, here ``V_{\beta}`` is Bernstein basis Vendermonde matrix.    

We also expand both ``q(x)`` and ``g(x)`` in Bernstein of the same degree:

``\vec{g} = V_{\beta}\cdot \vec{a}_{lower}``

``\vec{q} = V_{\beta}\cdot \vec{a}_{upper}``

Both ``\vec{a}_{lower}``  and  ``\vec{a}_{upper}`` are known vectors (as far as ``g(x)`` and ``q(x)`` are known functions)

"""

# ╔═╡ 9ddf624d-1e9f-41d9-9dd3-743a2730e89a
md"""
Original problem can now be reformulated in Bernstein basis:

``\vec{a^*} = argmin(\Phi(\vec{a}))``

"""

# ╔═╡ a842e84c-2566-4ef2-8801-2dcd489e2f37
md"""
Bernstein polynomial optimization  with constraints

"""

# ╔═╡ 616b871e-3229-4cb2-a60e-e9044485a330
@bind  a_f PlutoUI.combine() do Child
	md"""
	Coefficients of f(x) function
	
	``a_0`` = $(
		Child(Slider(-3:0.001:3,default=0.3,show_value = true))
	) \
	``a_1`` = $(
		Child(Slider(-3:0.001:3,default=0.1,show_value = true))
	)\
	``a_2`` = $(
		Child(Slider(-3:0.001:3,default=0.2,show_value = true))
	)\
	``a_3`` = $(
		Child(Slider(-3:0.001:3,default=0.2,show_value = true))
	)
	"""
end

# ╔═╡ 32dccd18-3140-446d-b997-8fd01d0dd359
md"Add noise $(@bind demo_data_noise Slider(0.0:1e-2:1, default =  0.8, show_value = true))"

# ╔═╡ d66a01ba-6040-4935-b36b-065015a0510c
@bind  a_lb PlutoUI.combine() do Child
	md"""
	Coefficients of f(x) function lower boundary g(x), ``g(x) \le f(x) ``
	
	``a_0`` = $(
		Child(Slider(-3:0.001:3,default=0.1,show_value = true))
	) \
	``a_1`` = $(
		Child(Slider(-3:0.001:3,default=0.1,show_value = true))
	)\
	``a_2`` = $(
		Child(Slider(-3:0.001:3,default=0.1,show_value = true))
	)\
	``a_3`` = $(
		Child(Slider(-3:0.001:3,default=0.1,show_value = true))
	)
	"""
end

# ╔═╡ 68e6ca68-e895-4f4f-a2a0-945422827e22
@bind  a_ub PlutoUI.combine() do Child
	md"""
	Coefficients of f(x) function upper boundary
	q(x), ``f(x) \le q(x)``
	
	``a_0`` = $(
		Child(Slider(-3:0.001:3,default=0.8,show_value = true))
	) \
	``a_1`` = $(
		Child(Slider(-3:0.001:3,default=0.8,show_value = true))
	)\
	``a_2`` = $(
		Child(Slider(-3:0.001:3,default=0.8,show_value = true))
	)\
	``a_3`` = $(
		Child(Slider(-3:0.001:3,default=0.8,show_value = true))
	)
	"""
end

# ╔═╡ c8f1e2f2-cefb-4009-8c08-2574e1e6846a
md"Use constraints $(@bind is_constraint CheckBox(false))"

# ╔═╡ 46a3593e-9653-49e8-8f11-b39b61176897
begin
	Bound_Type = ScaledPolynomials.BernsteinSymPoly{length(a_lb),Float64}
	v_data = ScaledPolynomials.VanderMatrix(x,Bound_Type())
	a_f_vect = [a_f...]
	y_bern_noisy = (1.0 .+ demo_data_noise*rand(NPOINTS)).*(v_data*a_f_vect);
	
	
	bound_vander_mat = ScaledPolynomials.VanderMatrix(x,Bound_Type())
	a_lb_vect = [a_lb...]
	a_ub_vect = [a_ub...]
end;

# ╔═╡ 5cfef29e-4687-46b2-89d0-97c7579e198e
md"""
	Select optimizer $(@bind optimizer Select([ParticleSwarm()=>"Particle swarm",NelderMead()=>"Nelder-Mead",LBFGS()=>"LBFGS"] ))
	"""

# ╔═╡ 106d13d1-5310-4a9f-a688-778351c56a9f
begin 
	fun_optim = (x,_)  -> sum( t->^(t,2) , y_bern_noisy .- bound_vander_mat*x)
	fun_opt = OptimizationFunction(fun_optim , NoAutoDiff()) 
	starting_vector =0.5*(  a_ub_vect .+ a_lb_vect )
	probl = if is_constraint
	 OptimizationProblem(fun_opt, 
                            starting_vector,
                            lb = a_lb_vect, # both min and max of emissivity should 
                            ub = a_ub_vect) # both min and max should be higher than 
	else
			 OptimizationProblem(fun_opt, 
                            starting_vector) 
	end
	a_solve = solve(probl,optimizer)
	
end

# ╔═╡ 76c60a39-9d68-4765-b698-7c1943572be8
begin
	pp_b = scatter(x,y_bern_noisy,label="data",legendcolumn=3) #1
	
	bern_coeffs_locations = ScaledPolynomials.bern_max_locations(bound_vander_mat)
	
	plot!(pp_b,x,bound_vander_mat*a_lb_vect,label="lower bound",linewidth=3,linecolor=:green)
	
	scatter!(pp_b,bern_coeffs_locations,a_lb_vect,markercolor=:green,label ="lb coeffs", markersize = 6)
	plot!(pp_b,x,bound_vander_mat*a_solve,linewidth=3,label = "fitting results")
	plot!(pp_b,x,bound_vander_mat*a_ub_vect,label="upper bound",linewidth=3,linecolor=:red)
	scatter!(pp_b,bern_coeffs_locations,a_ub_vect,markercolor=:red,label ="ub coeffs", markersize = 6)
	scatter!(pp_b,bern_coeffs_locations,a_solve,markercolor=:black,markerstype=:diamond,markersize = 8, label = "fitted aᵢ")
end

# ╔═╡ 6aea3077-0fd9-47ab-8da6-30ef3fc46bda
begin pretty_table(HTML,hcat(a_f_vect,a_lb_vect,a_solve,a_ub_vect) ,column_labels = ["Data coeffs", "Lower bound", "Fitted coeffs", "Upper bound"], title = "Constraint optimization")
end

# ╔═╡ 342c4189-1164-4c0f-a4c9-029c8720caea
if PolyType <: ScaledPolynomials.BernsteinPoly || PolyType <: ScaledPolynomials.BernsteinSymPoly
	ScaledPolynomials.bern_max(PolyType,1)
	ScaledPolynomials.bern_max_values(PolyType())
	ScaledPolynomials.bern_max_locations(PolyType())
end;

# ╔═╡ 8e492f7d-b959-4635-b524-feb48ba5f139
begin 
	NNN = 10
	XXX = SVector{NNN}(collect(range(-1.0,1.0,NNN)))
	V_stand = ScaledPolynomials.VanderMatrix(XXX,ScaledPolynomials.StandPoly{NNN,Float64}())
	V_leg = ScaledPolynomials.VanderMatrix(XXX,ScaledPolynomials.LegPoly{NNN,Float64}())
	V_bern2 = ScaledPolynomials.VanderMatrix(XXX,ScaledPolynomials.BernsteinSymPoly{NNN,Float64}())
	p_spectra = plot()
	for V in (V_stand,V_leg,V_bern2)
		plot!(p_spectra, svd(V.v).S, label = string(ScaledPolynomials.poly_name(V)),linewidth= 3,markersize=8,marker=:auto,markeralpha=0.5)
	end
	title!(p_spectra,"Singular values spectra for various polynomial bases")
	p_spectra
end

# ╔═╡ Cell order:
# ╟─d20e7502-fa03-4448-84f4-dc46acf52c9b
# ╟─14fe8399-7d59-4382-bc53-6e9ae4c05a8c
# ╟─f240500d-1198-49f7-b043-beea86a248c7
# ╟─c8102be0-6f8f-406e-9f18-45f462116602
# ╟─381fd867-f404-40e5-a270-ae98f50bf9b5
# ╟─1b15f3f8-fd8e-43c5-8c21-0b6fca139427
# ╟─81edb295-c20f-47a9-8a6f-d21e475df6a9
# ╟─0dbf180e-8b27-4be7-8747-e424f37e2e50
# ╟─a6381315-af21-4117-beed-6e3f7b8b8ca9
# ╟─c6dfdffe-8dd7-4264-9b75-c435c5bbf941
# ╟─3b475ddd-be05-4c56-9b28-68a808fc6e11
# ╟─e243649b-b920-4ca3-b2f3-6e9b5ada13c9
# ╟─7d6bf89c-4a52-4ed3-93ba-08c9f972c7a4
# ╟─52c7978a-c51b-40ac-9604-0485ff469141
# ╟─bf5df253-2d64-470b-8592-e93a6bcd144a
# ╟─eb75c9bd-d3f1-4cd2-bcf1-b53e6f7bed96
# ╟─1fa31f4b-7ed5-46fa-87ad-e5a4d22d59d6
# ╟─d130ca1e-94be-48b3-9c09-956a7a6b6bf4
# ╟─e9e28b59-36a2-406c-8f1e-488140aa9bc8
# ╟─9aad3160-fb56-4407-bdb7-f969d2b982a3
# ╟─54e44992-423e-4ebb-92e6-2af89b52e7f7
# ╟─bb55121f-9e67-446d-bf4f-4fc3dfe48062
# ╟─3280488d-efef-4833-ae37-6ea534f50df4
# ╟─facb5dae-3ee9-4380-8c92-0e94c93c84bc
# ╟─14ef768c-7039-4f34-862d-f58acf89e7d6
# ╟─0b00c857-aeb8-47ee-a9c2-b94fc32e9fb3
# ╟─56554cd8-e858-43d8-b6ef-7593d044920c
# ╟─132a1594-326c-4d48-88ce-b7756656b606
# ╟─c0418e2e-a6b6-4af5-a639-6d13952c88de
# ╟─7e1bcb20-a71a-4e9b-b797-dea6922f4cf8
# ╟─edeab52f-317e-42dc-840e-5bedece45dd8
# ╟─8c2bdb7c-aad4-4dd7-b6e2-97cf7dce0e56
# ╟─faaa4542-2066-49fc-b978-a6a45a4cca4e
# ╟─67918393-0efb-4c8a-a291-3d86f7853cce
# ╟─fd446c33-d439-439f-bd0a-f36770856cd2
# ╟─67ec5900-5ab4-4b8d-b9ac-7d34a01a7229
# ╟─4349710e-529b-4036-8986-6140292457c2
# ╟─9c73d7b0-547a-4577-8655-45a4e13df7a8
# ╟─7dfbac39-8e1b-4d88-830f-d18d94ed5e1b
# ╟─9ddf624d-1e9f-41d9-9dd3-743a2730e89a
# ╟─a842e84c-2566-4ef2-8801-2dcd489e2f37
# ╟─616b871e-3229-4cb2-a60e-e9044485a330
# ╟─32dccd18-3140-446d-b997-8fd01d0dd359
# ╟─d66a01ba-6040-4935-b36b-065015a0510c
# ╟─76c60a39-9d68-4765-b698-7c1943572be8
# ╟─68e6ca68-e895-4f4f-a2a0-945422827e22
# ╟─c8f1e2f2-cefb-4009-8c08-2574e1e6846a
# ╟─6aea3077-0fd9-47ab-8da6-30ef3fc46bda
# ╟─46a3593e-9653-49e8-8f11-b39b61176897
# ╟─5cfef29e-4687-46b2-89d0-97c7579e198e
# ╟─106d13d1-5310-4a9f-a688-778351c56a9f
# ╟─342c4189-1164-4c0f-a4c9-029c8720caea
# ╟─8e492f7d-b959-4635-b524-feb48ba5f139
