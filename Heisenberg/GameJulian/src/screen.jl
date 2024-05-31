struct Screen
    window
    renderer
    height::Int
    width::Int
    background::Union{ARGB, Ptr{SDL_Surface}}

    function Screen(name, w, h, background)
        win, renderer = makeWinRenderer(name, w, h)
        new(win, renderer, h, w, to_ARGB(background))
    end
end

#non ARGB colorant is converted to ARGB
#ARGB colorant is rerturned as is
# non colorant is returned as is (required since background is stored as an Union )
to_ARGB(c) = c
to_ARGB(c::ARGB) = c
to_ARGB(c::Colorant) = ARGB(c)


abstract type Geom end

"""
`Rect(x::Int,y::Int,w::Int,h::Int)`
`Rect(x::Tuple, y::Tuple)`

Creates an actor representing a rectangle.
"""
mutable struct Rect <: Geom
    x::Int
    y::Int
    w::Int
    h::Int
end
Rect(x::Tuple, y::Tuple) = Rect(x[1], x[2], y[1], y[2])

import Base:+
+(r::Rect, t::Tuple{T,T}) where T <: Number = Rect(Int(r.x+t[1]), Int(r.y+t[2]), r.h, r.w)

"""
`Line(x1::Int, y1::Int, x2::Int, y2::Int)`
`Line(x::Tuple, y::Tuple)`

Creates an actor representing a line. 
"""
mutable struct Line <: Geom  
    x1::Int
    y1::Int
    x2::Int
    y2::Int
end

mutable struct LineF <: Geom  
    x1::Float64
    y1::Float64
    x2::Float64
    y2::Float64
    c::Float64
end

Line(x::Tuple, y::Tuple) = Line(x[1], x[2], y[1], y[2])
LineF(x::Tuple, y::Tuple,c::Float64) = LineF(x[1], x[2], y[1], y[2],c)

mutable struct Triangle <: Geom
  p1::Vector{Int}
  p2::Vector{Int}
  p3::Vector{Int}

  function Triangle(p1::Vector{Int}, p2::Vector{Int}, p3::Vector{Int})
    if !(length(p1) == length(p2) == length(p3) == length(p3) == 2)
      error("Given vectors are of incompatible size.")
    end
    return new(p1, p2, p3)
  end

  function Triangle(x1::Int, y1::Int, x2::Int, y2::Int, x3::Int, y3::Int)
    new([x1; y1], [x2; y2], [x3; y3])
  end

  function Triangle(p1::Tuple, p2::Tuple, p3::Tuple)
    if !(length(p1) == length(p2) == length(p3) == length(p3) == 2)
      error("Given tuples are of incompatible size.")
    end
    return new([p1[1]; p1[2]], [p2[1]; p2[2]], [p3[1]; p3[2]])
  end
end

mutable struct Circle <: Geom
    x::Int
    y::Int
    r::Int
end


Base.convert(T::Type{SDL_Rect}, r::Rect) = SDL_Rect(Cint.((r.x, r.y, r.w, r.h))...)

function Base.setproperty!(s::Geom, p::Symbol, x)
    if hasfield(typeof(s), p)
        setfield!(s, p, Int(round(x)))
    else
        v = getPos(Val(p), s, x)
        setfield!(s, :x, Int(round(v[1])))
        setfield!(s, :y, Int(round(v[2])))
    end
end

function Base.getproperty(s::Geom, p::Symbol) 
    if hasfield(typeof(s), p)
        getfield(s, p)
    else
        v = getPos(Val(p), s)
        return v
    end
end

# A rect is stored as x/y/width/height. The following functions
# then calculate other metrics dynamically. Some metrics refer to 
# two dimensions (topright); they return tuples. Some other refer to 
# a single dimension (top); they return scalars
# The functions are called from `getproperty` for Rect
getPos(::Val{:left}, s::Rect) = s.x
getPos(::Val{:right}, s::Rect) = s.x+s.w
getPos(::Val{:top}, s::Rect) = s.y
getPos(::Val{:bottom}, s::Rect) = s.y+s.h
getPos(::Val{:pos}, s::Rect) = getPos(Val(:topleft), s)
getPos(::Val{:topleft}, s::Rect) = (s.x, s.y)
getPos(::Val{:topright}, s::Rect) = (s.x+s.w, s.y)
getPos(::Val{:bottomleft}, s::Rect) = (s.x, s.y+s.h)
getPos(::Val{:bottomright}, s::Rect) = (s.x+s.w, s.y+s.h)
getPos(::Val{:center}, s::Rect) = (s.x+s.w/2, s.y+s.h/2)
getPos(::Val{:centerx}, s::Rect) = s.x+s.w/2
getPos(::Val{:centery}, s::Rect) =  s.y+s.h/2
getPos(::Val{:centerleft}, s::Rect) = (s.x, s.y+s.h/2)
getPos(::Val{:centerright}, s::Rect) = (s.x+s.w, s.y+s.h/2)
getPos(::Val{:bottomcenter}, s::Rect) = (s.x+s.w/2, s.y+s.h)
getPos(::Val{:topcenter}, s::Rect) = (s.x+s.w/2, s.y)


# The following functions are used to postion a rectangle using various metrics. 
# They essentially return the x and y position, given other metrics (eg. bottomleft)
# These functions are called from `setproperty` for Rect
getPos(X::Val, s::Geom, v...) = nothing
getPos(X::Val, s::Geom, v::Tuple) = getPos(X, s, v[1], v[2])

getPos(::Val{:left}, s::Rect, v) = (v, s.y)
getPos(::Val{:right}, s::Rect, v) = (v-s.w, s.y)
getPos(::Val{:top}, s::Rect, v) = (s.x, v)
getPos(::Val{:bottom}, s::Rect, v) = (s.x, v-s.h)
getPos(::Val{:pos}, s::Rect, u, v) = getPos(Val(:topleft), s, u, v)
getPos(::Val{:topleft}, s::Rect, u, v) = (u, v)
getPos(::Val{:topright}, s::Rect, u, v) = (u-s.w, v)
getPos(::Val{:bottomleft}, s::Rect, u, v) = (u, v-s.h)
getPos(::Val{:bottomright}, s::Rect, u, v) = (u-s.w, v-s.h)
getPos(::Val{:center}, s::Rect, u, v) = (u-s.w/2, v-s.h/2)
getPos(::Val{:centerx}, s::Rect, u) = (u-s.w/2, s.y)
getPos(::Val{:centery}, s::Rect, v) = (s.x, v-s.h/2)
getPos(::Val{:centerleft}, s::Rect, u, v) = (u, v-s.h/2)
getPos(::Val{:centerright}, s::Rect, u, v) = (u-s.w, v-s.h/2)
getPos(::Val{:bottomcenter}, s::Rect, u, v) = (u-s.w/2, v-s.h)
getPos(::Val{:topcenter}, s::Rect, u, v) = (u-s.w/2, v)


getPos(::Val{:center}, s::Triangle) = (s.p1 + s.p2 + s.p3) / 3
getPos(::Val{:left}, s::Triangle) = min(s.p1[1], s.p2[1], s.p3[1])
getPos(::Val{:right}, s::Triangle) = max(s.p1[1], s.p2[1], s.p3[1])
getPos(::Val{:top}, s::Triangle) = max(s.p1[2], s.p2[2], s.p3[2])
getPos(::Val{:bottom}, s::Triangle) = min(s.p1[2], s.p2[2], s.p3[2])
getPos(::Val{:centerx}, s::Triangle) = (s.p1[1] + s.p2[1] + s.p3[1]) / 3
getPos(::Val{:centery}, s::Triangle) = (s.p1[2] + s.p2[2] + s.p3[2]) / 3

getPos(::Val{:center}, s::Circle, u, v) = (u, v)
getPos(::Val{:top}, s::Circle, v) = (s.x, v-s.r)
getPos(::Val{:bottom}, s::Circle, v) = (s.x, v+s.r)
getPos(::Val{:left}, s::Circle, u) = (u-s.r, s.y)
getPos(::Val{:right}, s::Circle, u) = (u+s.r, s.y)
getPos(::Val{:centerx}, s::Circle, u) = (u, s.y)
getPos(::Val{:centery}, s::Circle, v) = (s.x, v)

getPos(::Val{:center}, s::Circle) = (s.x, s.y)
getPos(::Val{:top}, s::Circle) = s.y-s.r
getPos(::Val{:bottom}, s::Circle) = s.y+s.r
getPos(::Val{:left}, s::Circle) = s.x-s.r
getPos(::Val{:right}, s::Circle) = s.x+s.r
getPos(::Val{:centerx}, s::Circle) = s.x
getPos(::Val{:centery}, s::Circle) = s.y

function clear(s::Screen)
    fill(s, s.background)
end

clear() = clear(game[].screen)

function Base.fill(s::Screen, c::Colorant)
    SDL_SetRenderDrawColor(
        s.renderer,
        sdl_colors(c)...,
    )
    SDL_RenderClear(s.renderer)
end

function Base.fill(s::Screen, sf::Ptr{SDL_Surface}) 
    texture = SDL_CreateTextureFromSurface(s.renderer, sf)
    SDL_RenderCopy(s.renderer, texture, C_NULL, C_NULL)
    SDL_DestroyTexture(texture)
end

draw(l::T, args...; kv...) where T <: Geom = draw(game[].screen, l, args...; kv...)

function draw(s::Screen, l::LineF, c::Colorant=colorant"red"; jbool=true)
    SDL_SetRenderDrawColor(
        s.renderer,
        sdl_colors(c)...,
    )
    if jbool
        #print("plotting something")
        dx = l.x1 - l.x2
        dy = l.y1 - l.y2
        norm = sqrt(dx^2+dy^2)
        dx /= norm
        dy /= norm
        px = 4 * (-dy) 
        #px = 4 
        py = 4 * (dx)
        #py = 4
        #t2 = SDL_CreateTexture(s.renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET,trunc(Int,px),trunc(Int,px));
        #SDL_SetTextureColorMod(t2,255, 0, 0);
        #SDL_SetTextureBlendMode(t2, SDL_BLENDMODE_NONE);
        #SDL_RenderCopy(s.renderer, t2, C_NULL, C_NULL)
        #SDL_DestroyTexture(t2)

        r = trunc( UInt8, 255 + ((0 - 255) * l.c))
        #g = 0
        b = trunc( UInt8,0 + ((255 - 0) * l.c))
        col =  SDL_Color(r,0,b,255)
           # r::Uint8
           # g::Uint8
           # b::Uint8
           # a::Uint8

        #col =  SDL_Color(255,0,0,255)
        #println(col)
        #println(SDL_Color(sdl_colors(colorant"red")...))
        #SDL_Color(sdl_colors(colorant"red")...)

        vertex_1 =  SDL_Vertex(SDL_FPoint(l.x1+px, l.y1+py),col,SDL_FPoint(1,1))
        vertex_2 =  SDL_Vertex(SDL_FPoint(l.x2+px, l.y2+py),col,SDL_FPoint(1,1))
        vertex_3 =  SDL_Vertex(SDL_FPoint(l.x1-px, l.y1-py),col,SDL_FPoint(1,1))
        vertex_4 =  SDL_Vertex(SDL_FPoint(l.x2-px, l.y2-py),col,SDL_FPoint(1,1))
        #vertex_4 =  SDL_Vertex(SDL_FPoint(l.x2-px, l.y2-py),SDL_Color(sdl_colors(colorant"red")...),SDL_FPoint(1,1))
       
      
        
        #vertex_1 = [[l.x1+px, l.y1+py], sdl_colors(colorant"red")..., C_NULL]
        #vertex_1 = ([100, 100], sdl_colors(colorant"red")..., [1,1])
        #vertex_2 = [[l.x2+px, l.y2+py], sdl_colors(colorant"red")..., C_NULL]
        #vertex_2 = ([200, 100], sdl_colors(colorant"red")..., [1,1])
        
        #vertex_3 = [[l.x1-px, l.y1-py], sdl_colors(colorant"red")..., C_NULL]
        #vertex_3 = ([100, 200], sdl_colors(colorant"red")..., [1,1])
        #vertex_4 = [[l.x2-px, l.y2-py], sdl_colors(colorant"red")..., C_NULL]

        #vertices = [vertex_1,vertex_2,vertex_3,vertex_4]
        vertices_1 = [vertex_1,vertex_2,vertex_3]
        vertices_2 = [vertex_2,vertex_3,vertex_4]

         #text = SDL_CreateTextureFromSurface(s.renderer, sf)
         #renderer::Core.Any, texture::Core.Any, vertices::Core.Any, num_vertices::Core.Any, indices::Core.Any, num_indices::Core.Any
        #SDL_RenderGeometry( s.renderer,nullptr, vertices,4,C_NULL,0)
        
        #SDL_RenderClear( s.renderer )
        SDL_RenderGeometry( s.renderer, C_NULL, vertices_1 , 3, C_NULL, 0 )
        SDL_RenderGeometry( s.renderer, C_NULL, vertices_2 , 3, C_NULL, 0 )
        #SDL_RenderPresent( s.renderer )
        #SDL_RenderDrawLine(s.renderer, Cint.((l.x1, l.y1, l.x2, l.y2))...)
    else
        SDL_RenderDrawLine(s.renderer, Cint.((l.x1, l.y1, l.x2, l.y2))...)
    end
end

function draw(s::Screen, r::Rect, c::Colorant=colorant"black"; fill=false)
    SDL_SetRenderDrawColor(
        s.renderer,
        sdl_colors(c)...,
    )
    sr = convert(SDL_Rect, r)
    if !fill
        SDL_RenderDrawRect(s.renderer, Ref(sr))
    else
        SDL_RenderFillRect(s.renderer, Ref(sr))
    end
end

sdl_colors(c::Colorant) = sdl_colors(convert(ARGB{Colors.FixedPointNumbers.Normed{UInt8,8}}, c))
sdl_colors(c::ARGB) = Int.(reinterpret.((red(c), green(c), blue(c), alpha(c))))

function draw(s::Screen, tr::Triangle, c::Colorant=colorant"black"; fill=false)
  p1, p2, p3 = Cint.(tr.p1), Cint.(tr.p2), Cint.(tr.p3)
  SDL_SetRenderDrawColor(s.renderer, sdl_colors(c)...)
  SDL_RenderDrawLines(s.renderer, [p1; p2; p3; p1], Cint(4))

  ymax = max(p1[2], p2[2], p3[2])
  ymin = min(p1[2], p2[2], p3[2])
  if fill && ymin != ymax
    # Set q1, q2 and q3 in descending order of y-value
    q1 = (p1[2] != ymax != p2[2]) * p3 +
        (p2[2] != ymax != p3[2]) * p1 +
        (p3[2] != ymax != p1[2]) * p2 +
        (p1[2] == p2[2] == ymax) * p2 +
        (p2[2] == p3[2] == ymax) * p3 +
        (p3[2] == p1[2] == ymax) * p1
    q3 = (p1[2] != ymin != p2[2]) * p3 +
        (p2[2] != ymin != p3[2]) * p1 +
        (p3[2] != ymin != p1[2]) * p2 +
        (p1[2] == p2[2] == ymin) * p2 +
        (p2[2] == p3[2] == ymin) * p3 +
        (p3[2] == p1[2] == ymin) * p1
    q2 = ((q1 == p1 && q3 == p3) || (q1 == p3 && q3 == p1)) * p2 +
        ((q1 == p1 && q3 == p2) || (q1 == p2 && q3 == p1)) * p3 +
        ((q1 == p2 && q3 == p3) || (q1 == p3 && q3 == p2)) * p1

    n = q1[2] - q2[2]
    x0 = q1[1] + (q2[2] - q1[2]) / (q3[2] - q1[2]) * (q3[1] - q1[1])
    for j = Cint(0):n-Cint(1)
      r1 = [round(Cint, q2[1] + j / n * (q1[1] - q2[1])); q2[2] + j]
      r2 = [round(Cint, x0 + j / n * (q1[1] - x0)); q2[2] + j]
      SDL_RenderDrawLines(s.renderer, [r1; r2], Cint(2))
    end
    n = q2[2] - q3[2]
    for j = Cint(1):n-Cint(1)
      r1 = [round(Cint, q2[1] + j / n * (q3[1] - q2[1])); q2[2] - j]
      r2 = [round(Cint, x0 + j / n * (q3[1] - x0)); q2[2] - j]
      SDL_RenderDrawLines(s.renderer, [r1; r2], Cint(2))
    end
  end
end

# improved circle drawing algorithm. slower but fills completely. needs optimization
function draw(s::Screen, circle::Circle, c::Colorant=colorant"black"; fill=false)
    # define the center and needed sides of circle
    centerX = Cint(circle.x)
    centerY = Cint(circle.y)
    int_rad = Cint(circle.r)
    left = centerX - int_rad
    top = centerY - int_rad

    SDL_SetRenderDrawColor(
        s.renderer,
        sdl_colors(c)...,
    )

    # we consider a grid with sides equal to the circle's diameter
    for x in left:centerX
        for y in top:centerY

            # for each pixel in the top left quadrant of the grid we measure the distance from the center.
            dist = sqrt( (centerX - x)^2 + (centerY - y)^2 )

            # if it is close to the circle's radius it and all associated points in the other quadrants are colored in.
            if (dist <= circle.r + 0.5 && dist >= circle.r - 0.5)
                rel_x = centerX - x
                rel_y = centerY - y

                quad1 = (x              , y              )
                quad2 = (centerX + rel_x, y              )
                quad3 = (x              , centerY + rel_y)
                quad4 = (quad2[1]       , quad3[2]       )

                SDL_RenderDrawPoint(s.renderer, quad1[1], quad1[2])
                SDL_RenderDrawPoint(s.renderer, quad2[1], quad2[2])
                SDL_RenderDrawPoint(s.renderer, quad3[1], quad3[2])
                SDL_RenderDrawPoint(s.renderer, quad4[1], quad4[2])

                # if we are told to fill in the circle we draw lines between all of the quadrants to completely fill the circle
                if (fill == true)
                    SDL_RenderDrawLine(s.renderer, quad1[1], quad1[2], quad2[1], quad2[2])
                    SDL_RenderDrawLine(s.renderer, quad2[1], quad2[2], quad4[1], quad4[2])
                    SDL_RenderDrawLine(s.renderer, quad4[1], quad4[2], quad3[1], quad3[2])
                    SDL_RenderDrawLine(s.renderer, quad3[1], quad3[2], quad1[1], quad1[2])
                end
            end

        end
    end

end

rect(x::Rect) = x
rect(x::Circle) = Rect(x.left, x.top, 2*x.r, 2*x.r)
