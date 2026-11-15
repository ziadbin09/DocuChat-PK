# Merchant API Reference

*Kohsar Couriers is a fictional company. This document is sample data for DocuChat.*

## Authentication

Every request must send an API key in the `X-KC-Key` header. Keys are created in the merchant dashboard and can be rotated at any time. A rotated key keeps working for 24 hours so integrations can switch over without downtime.

## Rate limits

The standard plan allows 120 requests per minute. The enterprise plan allows 1,200 requests per minute. When the limit is exceeded the API returns HTTP 429 with error code E-2009 and a `Retry-After` header.

## Error codes

E-1001 means the API key is missing or invalid. E-1004 means the key has been revoked. E-2009 means too many requests. E-3002 means the destination postcode is outside the delivery network. E-4012 means the parcel weight is missing or is above the 30 kg limit. E-4020 means the cash on delivery amount is above the 200,000 PKR ceiling. E-5000 is an unexpected server error and is safe to retry.

## Webhooks

Webhooks notify your server when a parcel changes status. A failed webhook is retried 6 times with exponential backoff over 24 hours. Each webhook is signed with HMAC-SHA256 in the `X-KC-Signature` header, and your server should reject payloads whose signature does not match.

## Bulk booking

The bulk endpoint accepts up to 500 shipments in one request. Bookings are processed asynchronously, and the response contains a batch ID that can be polled for results.

## Sandbox

The sandbox environment lives at sandbox.kohsar.example and never creates real pickups. Sandbox data is wiped every Sunday at midnight.
