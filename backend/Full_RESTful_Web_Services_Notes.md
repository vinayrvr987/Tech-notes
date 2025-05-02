
# RESTful Web Services Notes

## Chapters Set 1

### 00:00 - Introduction
- REST is an architectural style for designing networked applications.
- It uses a stateless, client-server communication protocol — usually HTTP.

### 00:29 - How are REST APIs Stateless?
- Each request from a client to server must contain all the information needed to understand and process the request.
- Server does not store anything about the latest client request.

### 01:36 - Explain the HTTP Methods
| Method | Description |
|--------|-------------|
| GET    | Retrieve data |
| POST   | Submit data |
| PUT    | Update data (replace) |
| PATCH  | Update data (partial) |
| DELETE | Remove data |

### 02:09 - Explain the HTTP Codes
| Code | Meaning |
|------|---------|
| 200  | OK |
| 201  | Created |
| 400  | Bad Request |
| 401  | Unauthorized |
| 404  | Not Found |
| 500  | Internal Server Error |

### 02:32 - What is a URI?
- URI (Uniform Resource Identifier) uniquely identifies a resource on the web.

### 02:58 - Best Practices in Making URI for RESTful Web Services
- Use nouns: `/users`, `/products`
- Use plural names: `/users` not `/user`
- Use HTTP methods for actions: `GET /users`, `POST /users`

### 03:30 - Differences Between REST and SOAP
| Feature | REST | SOAP |
|---------|------|------|
| Protocol | HTTP | XML over various protocols |
| Format | JSON, XML | XML only |
| Lightweight | Yes | No |
| Performance | Faster | Slower |

### 04:17 - Differences Between REST and AJAX
- **REST** is an architectural style.
- **AJAX** is a technique to send/receive data asynchronously using JavaScript.

### 04:58 - Tools to Develop and Test REST APIs
- Postman, Insomnia, Swagger UI, curl, HTTPie

### 05:29 - Real-World Examples of REST APIs
- GitHub API, Twitter API, Google Maps API, OpenWeather API

### 05:59 - Pros and Cons of RESTful Web Services
#### Pros:
- Scalable, Stateless, Cacheable, Platform Independent
#### Cons:
- No standard rules (compared to SOAP), May be less secure if not implemented properly

---

## Chapters Set 2

### 00:00 - Introduction
- RESTful services allow interoperability between systems using HTTP methods.

### 00:30 - What is the Difference Between PUT, POST, and PATCH?
| Method | Purpose                  | Idempotent |
|--------|--------------------------|------------|
| PUT    | Replace entire resource  | ✅ Yes |
| POST   | Create new resource      | ❌ No  |
| PATCH  | Partial update           | ❌ No  |

### 01:22 - What is a Payload in the Context of a REST API?
- Payload is the data sent with the HTTP request (usually in the body).

### 01:47 - What is a REST Message?
- A REST message is an HTTP request or response consisting of:
  - Headers
  - Body (Payload)
  - HTTP method
  - URI

### 02:11 - What are the Core Components of an HTTP Request?
- Request Line (method + URI + version)
- Headers
- Body (optional)

### 02:59 - What are the Core Components of an HTTP Response?
- Status Line (version + code + reason)
- Headers
- Body (data or error message)

### 03:49 - What is an Idempotent Method and Why Are They Important?
- Idempotent methods give the same result no matter how many times they're executed.
- Helpful in retries without unintended effects.

### 04:32 - What's the Difference Between Idempotent and Safe HTTP Methods?
| Property | Safe | Idempotent |
|----------|------|------------|
| Does not modify resources | ✅ | ❌ |
| Can be repeated without side-effects | ✅ | ✅ |

### 04:51 - Explain Caching in a RESTful Architecture
- Caching stores responses to avoid repetitive processing.
- Uses headers like `Cache-Control`, `ETag`, `Expires`.

### 05:21 - Best Practices in Developing a RESTful Web Service
- Use nouns for resources
- Use proper HTTP methods and status codes
- Stateless interactions
- Version APIs (`/api/v1/`)
- Secure endpoints with authentication

